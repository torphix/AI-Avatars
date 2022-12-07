import torch
import torch.nn as nn


class SPADENorm(nn.Module):
    def __init__(self, in_d):
        self.norm = nn.BatchNorm2d(in_d)
        self.beta = nn.Conv2d()

