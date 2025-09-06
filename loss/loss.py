import torch
import torch.nn as nn
import numpy as np
from loss.sub_loss import *

class loss_used(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.loss_cfg = hp.train.loss
        self.name = self.loss_cfg.name

    def forward(self, output: dict, GT, GT_grad):
        inf = output['output']
        loss = 0
        loss_state = {}
        for n in self.name:
            if n == 'grad':
                inf_grad = output['output_grad']
                sub_loss_fn = LOSS_REGISTRY.get('l2')(reduction = self.loss_cfg[n]['reduction'])
                sub_loss = sub_loss_fn(inf_grad / GT_grad.max(), GT_grad / GT_grad.max()).mean(2)
            else:
                sub_loss_fn = LOSS_REGISTRY.get(str(n).lower())(reduction = self.loss_cfg[n]['reduction'])
                sub_loss = sub_loss_fn(inf, GT)

            loss = loss + self.loss_cfg[n]['weight'] * (sub_loss).mean() # * sub_loss_weight ** 2.
            loss_state.update({str(n): 1 - sub_loss.mean().item() if n == 'ssim' else sub_loss.mean().item()})

        return loss, loss_state








