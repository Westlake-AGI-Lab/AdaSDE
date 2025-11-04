import torch
from torch_utils import persistence
from torch_utils import distributed as dist
import solvers
from solver_utils import get_schedule
from inception import compute_inception_mse_loss
from inception import InceptionFeatureExtractor
#----------------------------------------------------------------------------

def get_solver_fn(solver_name):
    if solver_name == 'adasde':
        solver_fn = solvers.adasde_sampler
    elif solver_name == 'ipndm':
        solver_fn = solvers.ipndm_sampler
    elif solver_name == 'dpm':
        solver_fn = solvers.dpm_sampler
    elif solver_name == 'heun':
        solver_fn = solvers.heun_sampler
    elif solver_name == 'adasde_parallel':
        solver_fn = solvers.adasde_parallel_sampler
    else:
        raise ValueError("Got wrong solver name {}".format(solver_name))
    return solver_fn

# ---------------------------------------------------------------------------
@persistence.persistent_class
class adasde_loss:
    def __init__(
        self, num_steps=None, sampler_stu=None, sampler_tea=None, M=None, 
        schedule_type=None, schedule_rho=None, afs=False, max_order=None, 
        sigma_min=None, sigma_max=None, predict_x0=True, lower_order_final=True,
    ):
        self.num_steps = num_steps
        self.solver_stu = get_solver_fn(sampler_stu)
        self.solver_tea = get_solver_fn(sampler_tea)
        self.M = M
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.max_order = max_order
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        
        self.num_steps_teacher = None
        self.tea_slice = None           # a list to extract the intermediate outputs of teacher sampling trajectory
        self.t_steps = None             # baseline time schedule for student
        self.buffer_model = []          # a list to save the history model outputs
        self.buffer_t = []              # a list to save the history time steps
        self.lpips = None

    def __call__(self, predictor, net, tensor_in, labels=None, step_idx=None, teacher_out=None, condition=None, unconditional_condition=None, dataset=None):
        step_idx = torch.tensor([step_idx]).reshape(1,)
        t_cur = self.t_steps[step_idx].to(tensor_in.device)
        t_next = self.t_steps[step_idx + 1].to(tensor_in.device)

        if step_idx == 0:
            self.buffer_model = []
            self.buffer_t = []

        # Student steps.
        student_out, buffer_model, buffer_t, r_s, scale_dir_s, scale_time_s, weight_s, gamma_s = self.solver_stu(
            net, 
            tensor_in / t_cur, 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition,
            num_steps=2,
            sigma_min=t_next, 
            sigma_max=t_cur, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=self.afs, 
            denoise_to_zero=False, 
            return_inters=False, 
            predictor=predictor, 
            step_idx=step_idx, 
            train=True,
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
            buffer_model=self.buffer_model, 
            buffer_t=self.buffer_t, 
        )
        self.buffer_model = buffer_model
        self.buffer_t = buffer_t
        try:
            num_points = predictor.num_points
            alpha = predictor.alpha
        except:
            num_points = predictor.module.num_points
            alpha = predictor.module.alpha

        loss = (student_out - teacher_out) ** 2

        if step_idx == self.num_steps - 2:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            feature_extractor = InceptionFeatureExtractor(device=device)

            if dataset in ['lsun_bedroom_ldm','ms_coco']:
                student_out = net.model.differentiable_decode_first_stage(student_out)
                teacher_out = net.model.decode_first_stage(teacher_out)
            
            student_out = (student_out * 127.5 + 128).clip(0, 255)
            teacher_out = (teacher_out * 127.5 + 128).clip(0, 255)
            inception_loss = compute_inception_mse_loss(student_out, teacher_out, feature_extractor)
            loss = loss + alpha * inception_loss - loss
 
        str2print = f"Step: {step_idx.item()} | Loss: {torch.mean(torch.norm(loss, p=2, dim=(1, 2, 3))).item():8.4f} "
       
        p = getattr(predictor, 'module', predictor)

        # weights
        for i in range(num_points):
            weight = weight_s[:, i:i+1, :, :]
            str2print += f"| w{i}: {weight.mean().item():5.4f} "

        # r
        for i in range(num_points):
            r = r_s[:, i:i+1, :, :]
            str2print += f"| r{i}: {r.mean().item():5.4f} "

        # scale_time
        if bool(getattr(p, 'scale_time', 0)):
            for i in range(num_points):
                st = scale_time_s[:, i:i+1, :, :]
                str2print += f"| st{i}: {st.mean().item():5.4f} "

        # scale_dir
        if bool(getattr(p, 'scale_dir', 0)):
            for i in range(num_points):
                sd = scale_dir_s[:, i:i+1, :, :]
                str2print += f"| sd{i}: {sd.mean().item():5.4f} "

        has_gamma = float(getattr(p, 'gamma', 0)) > 0.0
        for i in range(num_points):
            g = gamma_s[:, i:i+1, :, :] if 'gamma_s' in locals() else torch.zeros_like(r_s[:, i:i+1, :, :])
            str2print += f"| gamma{i}: {g.mean().item():5.6f} "
        if not has_gamma:
            str2print += "| gamma_off "

        return loss, str2print, student_out

    
    def get_teacher_traj(self, net, tensor_in, labels=None, condition=None, unconditional_condition=None):
        if self.t_steps is None:
            self.t_steps = get_schedule(self.num_steps, self.sigma_min, self.sigma_max, schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, device=tensor_in.device, net=net)
        if self.tea_slice is None:
            self.num_steps_teacher = (self.M + 1) * (self.num_steps - 1) + 1
            self.tea_slice = [i * (self.M + 1) for i in range(1, self.num_steps)]
        
        # Teacher steps.
        teacher_traj = self.solver_tea(
            net, 
            tensor_in / self.t_steps[0], 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition, 
            num_steps=self.num_steps_teacher, 
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=False, 
            denoise_to_zero=False, 
            return_inters=True, 
            predictor=None, 
            train=False,
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
        )

        return teacher_traj[self.tea_slice]