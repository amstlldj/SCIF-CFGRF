import torch
import torch.nn.functional as F


class RectifiedFlow:
    def euler(self, x_t, v, dt):
        x_t = x_t + v * dt
        return x_t
        
    def create_flow(self, x_1, t, x_0=None):
        if x_0 is None:
           x_0 = torch.randn_like(x_1)
        t = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t = t * x_1 + (1 - t) * x_0
        return x_t, x_0
        
    def mse_loss(self, v, x_1, x_0):
        loss = F.mse_loss(x_1 - x_0, v)
        return loss
        