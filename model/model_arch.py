import torch.nn as nn
from model.model_used.models import *

class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, hp):
        super(Net_arch, self).__init__()
        self.model = MODEL_REGISTRY.get(str(hp.model.name).lower())(hp.model)
        self.LC_Layer = LightCorrectionLayer(hp.model)

    def forward(self):
        return self.model()

