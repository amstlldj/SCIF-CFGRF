from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from utils.RectifiedFlow import RectifiedFlow

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class RFDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Define Rectified Flow parameters (model for noise adjustment)
        self.rf = RectifiedFlow()  # Use the provided RF class

    def forward(self, x_0, labels=None, use_cfg=True):
        t = torch.rand(x_0.size(0), device=x_0.device)

        # Apply RF's create_flow to simulate the diffusion process
        x_t, x_noise = self.rf.create_flow(x_0, t.float())

        if use_cfg:
            x_t = torch.cat([x_t, x_t.clone()], dim=0) # ([batch_size,3,32,32]) --> ([2*batch_size,3,32,32])
            t = torch.cat([t, t.clone()], dim=0)
            labels = torch.cat([labels, torch.ones_like(labels)], dim=0)
            x_0 = torch.cat([x_0, x_0.clone()], dim=0)
            x_noise = torch.cat([x_noise, x_noise.clone()], dim=0)
        else:
            labels = None

        # Predict the Velocity Field with the model
        v_pred = self.model(x_t, t, labels) 

        # Compute the loss
        rf_gen_loss = self.rf.mse_loss(v_pred, x_0, x_noise)  # Calculate Rectified Flow MSE loss

        loss = torch.sum(rf_gen_loss)
        
        return loss, x_noise


class RFSampler(nn.Module):
    def __init__(self, model): 
        super().__init__()
        self.model = model
        # Define Rectified Flow model (to correct noise during reverse process)
        self.rf = RectifiedFlow()

    @torch.no_grad()
    def forward(self, x_t, batch_idx, labels=None, steps: int = 1, only_return_x_0: bool = True, interval: int = 1, cfg_scale:float = 1.0):
        x = [x_t]
        labels_list = []  # List to store labels at each sampling step
        # Euler
        dt = 1/steps
        # 提取第batch_idx批图像的标签条件labels_batch_idx
        if labels is not None:
            labels_batch_idx = labels[batch_idx]
            labels_batch_idx = labels_batch_idx.to(x_t.device)

        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                t = i * dt
                if labels is not None:
                    v_pred_uncond = self.model(x_t, t)
                    v_pred_cond = self.model(x_t, t, labels_batch_idx)
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
                else:
                    v_pred = self.model(x_t, t)
                
                x_t = self.rf.euler(x_t, v_pred, dt)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))
                    labels_list.append(labels_batch_idx)

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t, labels_batch_idx  # Final generated image and corresponding label
        else:
            return torch.stack(x, dim=1), torch.stack(labels_list, dim=1)  # Stack intermediate results

