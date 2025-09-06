import torch.nn as nn
import torch.nn.functional as F
from model.model_used.models import *

class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, hp, post_name = None):
        super(Net_arch, self).__init__()
        self.model = MODEL_REGISTRY.get(str(hp.model.name).lower() + '_' + post_name)(hp.model)
        self.LC_Layer = LightCorrectionLayer(hp.model)

    def forward(self):
        return self.model()

