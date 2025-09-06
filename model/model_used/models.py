import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import unfoldNd
from model.model_used.utils import *
from util.register import MODEL_REGISTRY
from math import pi

@MODEL_REGISTRY.register('nerf_opensourced')
class NeRF(nn.Module): # Hash + LR + PE
    def __init__(self, hp_net):

        super().__init__()
        self.skips = hp_net.skips
        self.full_res = hp_net.full_resolution
        self.hash_table_l = hp_net.hash_table_length
        self.hash_table_l_orig = hp_net.hash_table_length
        self.aberration = hp_net.aberration
        self.in_c = hp_net.in_features
        hidden_c = hp_net.hidden_features
        self.hidden_c = hidden_c
        hidden_l = hp_net.hidden_layers
        self.out_c = hp_net.out_features

        if not self.full_res:
            self.hash_table_l = list(np.around(np.array(self.hash_table_l) * np.array([hp_net.res_ratio_xy, hp_net.res_ratio_xy, hp_net.res_ratio_z])).astype(np.int16))

        self.table_real = nn.Parameter(1e-4 * (torch.rand(self.hash_table_l + [self.in_c]) * 2), requires_grad = True)

        if self.aberration:
            table_aberation_xy = list(np.array(hp_net.hash_table_length[:2]) // 4)
            self.table_aberration = nn.Parameter(1e-4 * (torch.rand(table_aberation_xy + [self.in_c]) * 2 - 1), requires_grad = True)
        self.input_features = self.in_c

        ''' layers '''
        self.net_layers_real = [LeakyReluLayer(self.input_features, hidden_c)] + \
                               [LeakyReluLayer(hidden_c, hidden_c) if i not in self.skips else
                                LeakyReluLayer(hidden_c + self.input_features, hidden_c) for i in range(hidden_l - 1)]
        self.output_layer_real = nn.Linear(hidden_c, self.out_c)
        self.init_weight_zero_bias_constant(self.output_layer_real)
        self.net_real = nn.ModuleList(self.net_layers_real)

        ''' aberration '''
        if self.aberration:
            self.net_layers_aberration = [LeakyReluLayer(self.in_c, hidden_c)] + \
                                         [LeakyReluLayer(hidden_c, hidden_c) if i not in self.skips else
                                          LeakyReluLayer(hidden_c + self.in_c, hidden_c) for i in range(hidden_l - 1)]
            self.output_layer_aberration = nn.Linear(hidden_c, 1)
            self.init_weight_zero(self.output_layer_aberration)
            self.net_aberration = nn.ModuleList(self.net_layers_aberration)


    def init_weight_zero(self, sub_net):
        nn.init.zeros_(sub_net.weight)
        nn.init.zeros_(sub_net.bias)

    def init_weight_zero_bias_constant(self, sub_net):
        nn.init.zeros_(sub_net.weight)
        nn.init.zeros_(sub_net.bias)

    def forward(self):
        H, W, D = self.hash_table_l_orig

        table_real = self.table_real.view(-1, self.in_c)

        ''' operation of RI '''
        x_real = table_real
        for i, l in enumerate(self.net_real):
            x_real = l(x_real)
            if i in self.skips:
                x_real = torch.cat([table_real, x_real], -1)
        if not self.full_res:
            orig_size = self.hash_table_l + [self.hidden_c]
            x_real = x_real.view(orig_size)
            x_real = F.interpolate(x_real[...,None].permute(3,4,0,1,2), [H, W, D], mode = 'trilinear',
                                   align_corners = True).permute(2,3,4,0,1)[...,0]
            x_real = x_real.view(-1, self.hidden_c)
        else:
            pass

        tensor_real = self.output_layer_real(x_real)

        output = F.tanh(tensor_real)

        ''' operation for aberration  '''
        if self.aberration:
            table_aberration = self.table_aberration.view(-1, self.in_c)
            aberration = table_aberration
            for i, l in enumerate(self.net_aberration):
                aberration = l(aberration)
                if i in self.skips:
                    aberration = torch.cat([table_aberration, aberration], -1)

            output_aberration = self.output_layer_aberration(aberration)
            output_aberration = F.tanh(output_aberration)

        return [output, output_aberration if self.aberration else None]

