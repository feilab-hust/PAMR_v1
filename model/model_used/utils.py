import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class LeakyReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype = torch.float, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype = dtype)
        self.leaky_relu = partial(F.relu, inplace = True)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.linear.bias, -0.05, 0.05)
            nn.init.kaiming_normal_(self.linear.weight, mode = 'fan_in', nonlinearity = 'relu')

    def forward(self, input):
        return self.leaky_relu(self.linear(input))

class LightCorrectionLayer(nn.Module):
    def __init__(self, hp_net):
        super().__init__()
        self.LEDs_num = hp_net.LED_used
        self.coeff = nn.Parameter(torch.ones(self.LEDs_num), requires_grad = True)

    def forward(self, x, idx):
        return x * self.coeff[idx]
