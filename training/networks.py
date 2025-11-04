import numpy as np
import torch
from torch_utils import persistence
import torch.nn as nn
from torch.nn.functional import silu
import torch.nn.functional as F
import math

@persistence.persistent_class
class adasde_predictor(torch.nn.Module):
    """
    Ensemble Parallel Directions
    """
    def __init__(
        self,
        num_points              = 2, # number of inter points
        dataset_name            = None,
        img_resolution          = None,
        num_steps               = None,
        sampler_tea             = None, 
        sampler_stu             = None, 
        M                       = None,
        guidance_type           = None,      
        guidance_rate           = None,
        schedule_type           = None,
        schedule_rho            = None,
        afs                     = False,
        scale_dir               = 0,
        scale_time              = 0,
        gamma                   = 0,
        max_order               = None,
        predict_x0              = True,
        lower_order_final       = True,
        fcn                     = False,
        alpha                   = 10,
        **kwargs
    ):
        super().__init__()
        assert sampler_stu in ['adasde']
        assert sampler_tea in ['dpm']
        assert scale_dir >= 0
        assert scale_time >= 0
        self.dataset_name = dataset_name
        self.img_resolution = img_resolution
        self.num_steps = num_steps
        self.sampler_stu = sampler_stu
        self.sampler_tea = sampler_tea
        self.M = M
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.scale_dir = scale_dir
        self.scale_time = scale_time
        self.gamma = gamma
        self.max_order = max_order
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.num_points = num_points
        self.fcn = fcn
        self.alpha = alpha
        self.gamma_min = 0.0002
        self.r_params = nn.Parameter(torch.randn(num_steps-1, num_points)) 
        self.scale_dir_params = nn.Parameter(torch.randn(num_steps-1, num_points))
        self.scale_time_params = nn.Parameter(torch.randn(num_steps-1, num_points))
        self.weight_s = nn.Parameter(torch.randn(num_steps-1, num_points))
        self.gamma_params = nn.Parameter(torch.randn(num_steps-1, num_points))

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, batch_size, step_idx, class_labels=None,):
        weight = self.weight_s[step_idx]
        weight = weight.repeat(batch_size, 1)
        weight = self.softmax(weight)
 
        r = self.r_params[step_idx]
        r = r.repeat(batch_size, 1)
        r = self.sigmoid(r)

        params = []


        if self.scale_dir:
            raw = self.scale_dir_params[step_idx].repeat(batch_size, 1)
            tau = 2.0 
            alpha = min(float(self.scale_dir), 0.99)  
            scale_dir = 1.0 + alpha * torch.tanh(raw / tau)
            params.append(scale_dir)

        if self.scale_time:
            raw = self.scale_time_params[step_idx].repeat(batch_size, 1)
            tau = 2.0
            alpha = min(float(self.scale_time), 0.99)
            scale_time = 1.0 + alpha * torch.tanh(raw / tau)
            params.append(scale_time)

        if self.gamma:
            raw  = self.gamma_params[step_idx].repeat(batch_size, 1)

            tau  = 2.0
            beta = 1.0
            p0   = 0.60

            p0_root = p0 ** (1.0 / beta)
            p0_root = min(max(p0_root, 1e-6), 1 - 1e-6)
            b = tau * math.log(p0_root / (1.0 - p0_root))

            s = torch.sigmoid((raw + b) / tau).pow(beta)          

            gmax = float(self.gamma)
            gmin = float(self.gamma_min)                          

            gamma = gmin + (gmax - gmin) * s                     
            params.append(gamma)

        params.append(weight)

        return (r, *params) if params else r
    
