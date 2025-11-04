import torch
from solver_utils import *
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor
from torch_utils import distributed as dist


#----------------------------------------------------------------------------
def get_adasde_prediction(predictor, step_idx: int, batch_size: int):
    """Query the AdaSDE predictor and normalize its outputs.

    The predictor is expected to return a tuple/list containing at least:
    - r (per-point interpolation exponent)
    - optionally: scale_dir, scale_time, gamma
    - weight (per-point weights)

    This helper reshapes everything to broadcastable 4D tensors
    with shape [-1, num_points, 1, 1], fills missing components with
    sensible defaults, and applies step-dependent adjustments.

    Args:
        predictor: AdaSDE predictor module or DDP-wrapped module (with .module).
        step_idx (int): Current sampling step index.
        batch_size (int): Batch size used to query the predictor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (r, scale_dir, scale_time, weight, gamma_t), each shaped
            [-1, num_points, 1, 1].
    """
    output = predictor(batch_size, step_idx)
    output_list = list(output) if isinstance(output, (tuple, list)) else [output]

    # Handle potential DDP wrapping and read predictor config.
    p = getattr(predictor, "module", predictor)
    num_points = p.num_points
    use_scale_dir = bool(getattr(p, "scale_dir", 0))
    use_scale_time = bool(getattr(p, "scale_time", 0))
    use_gamma = bool(getattr(p, "gamma", 0))
    num_steps = p.num_steps
    use_fcn = getattr(p, "fcn", False)

    # Required components: r and weight.
    r = output_list[0].reshape(-1, num_points, 1, 1)
    weight = output_list[-1].reshape(-1, num_points, 1, 1)

    # Defaults for optional components.
    zeros_like_r = torch.zeros_like(r)
    ones_like_r = torch.ones_like(r)
    scale_dir = ones_like_r
    scale_time = ones_like_r
    gamma_t = zeros_like_r  # Default gamma is 0.

    # Unpack optional outputs in order: scale_dir, scale_time, gamma.
    idx = 1
    if use_scale_dir:
        scale_dir = output_list[idx].reshape(-1, num_points, 1, 1)
        idx += 1
    if use_scale_time:
        scale_time = output_list[idx].reshape(-1, num_points, 1, 1)
        idx += 1
    if use_gamma:
        gamma_t = output_list[idx].reshape(-1, num_points, 1, 1)  # Already mapped to [0, gamma_max].

    # Step-dependent adjustments.
    if step_idx == 0:
        gamma_t = gamma_t.zero_()  # No gamma injection at the very first step.
    if step_idx == num_steps - 2 and use_fcn:
        # Force direction scaling to 1.0 at the penultimate step if FCN is enabled.
        scale_dir = scale_dir.clone()
        scale_dir.fill_(1.0)

    return r, scale_dir, scale_time, weight, gamma_t



#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):     # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------
def adasde_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    condition=None,
    unconditional_condition=None,
    num_steps=None,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type="time_uniform",
    schedule_rho=7,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    predictor=None,
    step_idx=None,
    train=False,
    verbose=False,
    **kwargs,
):
    """Sample with an AdaSDE-style predictor-corrector schedule.

    This routine performs iterative sampling using a learned AdaSDE predictor.
    It optionally injects noise inflation via gamma, supports AFS (analytic
    first step), and can return intermediate states for analysis/visualization.

    Args:
        net: Denoiser network. Called via `get_denoised`.
        latents (torch.Tensor): Initial latent tensor, shape [B, C, H, W].
        class_labels (torch.Tensor, optional): Class labels for conditional models.
        randn_like (Callable): Function to sample standard normal noise like a tensor.
        condition (Any, optional): Conditional input passed to the denoiser.
        unconditional_condition (Any, optional): Unconditional input for classifier-free guidance.
        num_steps (int, optional): Number of time steps.
        sigma_min (float): Minimum noise level for the schedule.
        sigma_max (float): Maximum noise level for the schedule.
        schedule_type (str): Schedule name passed to `get_schedule`.
        schedule_rho (float): Rho parameter for Karras-style schedules.
        afs (bool): If True, use analytic first step on the first iteration.
        denoise_to_zero (bool): If True, run a final denoise at the last t to t=0.
        return_inters (bool): If True, return stacked intermediates [T+1, B, C, H, W].
        predictor: AdaSDE predictor object used by `get_adasde_prediction`.
        step_idx (int, optional): External step index to query the predictor with.
        train (bool): Training mode flag (affects AFS gating and return type).
        verbose (bool): If True, print per-step predictor parameters.
        **kwargs: Ignored extra keyword arguments for API compatibility.

    Returns:
        If `return_inters` is True:
            torch.Tensor: Stacked intermediates with shape [T(+1), B, C, H, W].
        Else if `train` is True:
            Tuple[torch.Tensor, list, list, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (x_next, [], [], r_s, scale_dir_s, scale_time_s, weight_s, gamma_s)
        Else:
            Tuple[torch.Tensor, list]: (final_sample, list_of_states)

    Notes:
        - `get_schedule`, `get_denoised`, and `get_adasde_prediction` are expected
          to be available in the current namespace.
        - The predictor can return (r_s, scale_dir_s, scale_time_s, weight_s)
          or the same plus gamma_s. If gamma_s is absent, gamma is treated as zero.
    """
    assert predictor is not None

    # Time-step discretization.
    t_steps = get_schedule(
        num_steps,
        sigma_min,
        sigma_max,
        device=latents.device,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        net=net,
    )

    # Main sampling loop setup.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    x_list = [x_next]

    # Iterate over step pairs (t_cur -> t_next), indices 0..N-1.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Reshape scalars to broadcastable shapes.
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)

        # --- Fetch AdaSDE predictor parameters. ---
        pred_out = get_adasde_prediction(
            predictor,
            step_idx if step_idx is not None else i,
            batch_size=latents.shape[0],
        )

        if isinstance(pred_out, (tuple, list)) and len(pred_out) == 5:
            r_s, scale_dir_s, scale_time_s, weight_s, gamma_s = pred_out
        else:
            r_s, scale_dir_s, scale_time_s, weight_s = pred_out
            gamma_s = torch.zeros_like(r_s)  # No gamma provided: equivalent to no injection.

        # Per-step gamma (weighted average across points), shape [B, 1, 1, 1].
        gamma_step = (gamma_s * weight_s).sum(dim=1, keepdim=True)

        # --- Inflate noise at t_cur and replace (x_cur, t_cur) -> (x_hat, t_hat). ---
        # t_hat = (1 + gamma_step) * t_cur, with gamma_step ∈ [0, gamma_max].
        t_hat = (1.0 + gamma_step) * t_cur
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * randn_like(x_cur)

        # --- Compute current direction d_cur at (x_hat, t_hat). AFS also uses t_hat. ---
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_hat / ((1 + t_hat**2).sqrt())
        else:
            denoised_hat = get_denoised(
                net,
                x_hat,
                t_hat,
                class_labels=class_labels,
                condition=condition,
                unconditional_condition=unconditional_condition,
            )
            d_cur = (x_hat - denoised_hat) / t_hat

        dt = t_next - t_hat
        x_next = x_hat  # Start accumulation from x_hat.

        # Obtain number of parallel points from predictor.
        try:
            num_points = predictor.num_points
        except Exception:
            num_points = predictor.module.num_points

        if verbose:
            print_str = f"step {i}: |"
            for j in range(num_points):
                print_str += f"r{j}: {r_s[0,j,0,0]:.5f} "
                print_str += f"st{j}: {scale_time_s[0,j,0,0]:.5f} "
                print_str += f"sd{j}: {scale_dir_s[0,j,0,0]:.5f} "
                print_str += f"g{j}: {gamma_s[0,j,0,0]:.5f} "
                print_str += f"w{j}: {weight_s[0,j,:,:].mean().item():.5f} |"

            try:
                from torch_utils import distributed as dist
                # Print on rank 0 only to avoid multi-GPU spam.
                if dist.get_rank() == 0:
                    dist.print0(print_str)
            except Exception:
                print(print_str, flush=True)

        # --- Parallel midpoint updates (left endpoint changed to t_hat). ---
        for j in range(num_points):
            r = r_s[:, j : j + 1, :, :]
            scale_time = scale_time_s[:, j : j + 1, :, :]
            scale_dir = scale_dir_s[:, j : j + 1, :, :]
            w = weight_s[:, j : j + 1, :, :]

            # Geometric interpolation: t_mid ∈ [t_hat, t_next].
            t_mid = (t_next**r) * (t_hat ** (1 - r))

            # Extrapolate from (x_hat, t_hat) to t_mid.
            x_mid = x_hat + (t_mid - t_hat) * d_cur

            # Evaluate at scaled time: scale_time * t_mid.
            denoised_t_mid = get_denoised(
                net,
                x_mid,
                scale_time * t_mid,
                class_labels=class_labels,
                condition=condition,
                unconditional_condition=unconditional_condition,
            )
            d_mid = (x_mid - denoised_t_mid) / t_mid

            # Accumulate parallel directions; Δt uses (t_next - t_hat).
            x_next = x_next + w * scale_dir * dt * d_mid

        x_list.append(x_next)

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(
            net,
            x_next,
            t_next,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
        )
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r_s, scale_dir_s, scale_time_s, weight_s, gamma_s
    return x_next, x_list


#----------------------------------------------------------------------------
def dpm_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='time_uniform', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    inner_steps=3,  # New parameter for the number of inner steps
    r=0.5,
    **kwargs
):
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    x_list = []
    x_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Compute the inner step size
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device, schedule_type='polynomial', schedule_rho=7)
        for i, (t_c, t_n) in enumerate(zip(t_s[:-1],t_s[1:])):
            # Euler step.
            use_afs = (afs and i == 0)
            if use_afs:
                d_cur = x_cur / ((1 + t_c**2).sqrt())
            else:
                denoised = get_denoised(net, x_cur, t_c, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
                d_cur = (x_cur - denoised) / t_c
            t_mid = (t_n ** r) * (t_c ** (1 - r))
            x_next = x_cur + (t_mid - t_c) * d_cur

            # Apply 2nd order correction.
            denoised = get_denoised(net, x_next, t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            
            d_mid = (x_next - denoised) / t_mid
            x_cur = x_cur + (t_n - t_c) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)
        x_next = x_cur
        x_list.append(x_next)

        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list


#----------------------------------------------------------------------------

def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='time_uniform', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    inner_steps=3,  # New parameter for the number of inner steps
    r=0.5,
    **kwargs
):
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    x_list = []
    x_list.append(x_next)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Compute the inner step size
        t_s = get_schedule(inner_steps, t_next, t_cur, device=latents.device, schedule_type='polynomial', schedule_rho=7)
        for i, (t_c, t_n) in enumerate(zip(t_s[:-1],t_s[1:])):
            # Euler step.
            use_afs = (afs and i == 0)
            if use_afs:
                d_cur = x_cur / ((1 + t_c**2).sqrt())
            else:
                denoised = get_denoised(net, x_cur, t_c, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
                d_cur = (x_cur - denoised) / t_c
            x_next = x_cur + (t_n - t_c) * d_cur

            # Apply 2nd order correction.
            denoised = get_denoised(net, x_next, t_n, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            
            d_prime = (x_next - denoised) / t_n
            x_cur = x_cur + (t_n - t_c) * (0.5 * d_cur + 0.5 * d_prime)
        x_next = x_cur
        x_list.append(x_next)

        if return_inters:
            inters.append(x_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next, x_list

