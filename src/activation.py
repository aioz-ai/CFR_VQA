import torch
import torch.nn as nn

"""Define new activation for VQA"""

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return torch.mul(x, torch.sigmoid(x))