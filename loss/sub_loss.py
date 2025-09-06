import torch
import torch.nn as nn
from util.register import LOSS_REGISTRY
from util.ssim import SSIM

@LOSS_REGISTRY.register('smooth_l1')
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0, beta = 0.02):
        super().__init__()

        self.w = weight
        self.loss = nn.SmoothL1Loss(reduction = reduction, beta = beta)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('l2')
class MSELoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()

        self.w = weight
        self.loss = nn.MSELoss(reduction = reduction)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        loss = self.w * self.loss(inf, gt)
        return loss

@LOSS_REGISTRY.register('ssim')
class SSIMLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()
        self.w = weight
        self.loss = SSIM()

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        if len(inf.shape) == 4:
            inf = inf.permute(1,0,2,3)
            gt = gt.permute(1,0,2,3)
        loss = self.w * (1 - self.loss(inf, gt))
        return loss

@LOSS_REGISTRY.register('fft')
class FFTLoss(nn.Module):
    def __init__(self, reduction = 'none', weight = 1.0):
        super().__init__()
        self.w = weight
        # self.loss = nn.MSELoss(reduction = reduction)
        self.loss = nn.SmoothL1Loss(reduction = reduction, beta = 0.05)

    def forward(self, inf, gt):
        assert inf.shape == gt.shape, Exception('shape should be the same: inf: {}, gt: {}'.format(inf.shape, gt.shape))
        if len(inf.shape) == 4:
            inf = inf.permute(1,0,2,3)
            gt = gt.permute(1,0,2,3)

        inf_ft = torch.fft.fft2(inf)[...,1:, 1:]
        gt_ft = torch.fft.fft2(gt)[...,1:, 1:]
        loss = self.w * self.loss(torch.abs(inf_ft), torch.abs(gt_ft))
        return loss
