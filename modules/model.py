import torch
import torch.nn as nn
from modules.resnets import Resnet2d, Resblock2d
from .modules import AppearanceEncoder, WarpGenerator


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.expression_enc = Resnet2d(3, 1024, [3,4,6,3], Resblock2d)
        self.appearance_enc = AppearanceEncoder()
        self.warp_generator = WarpGenerator()

    def forward(self, src_x, drive_x, src_poses, drive_poses):
        # Extract poses
        # Extract features
        src_expr = self.expression_enc(src_x)
        drive_expr = self.expression_enc(drive_x)
        src_v, src_global = self.appearance_enc(src_x)
        # Prepare features
        src_features = (src_poses, src_expr + src_global)
        drive_features = (drive_poses, drive_expr + src_global)
        # Warp feature generation
        output_image = self.warp_generator(src_features, drive_features, src_v)
        return output_image
        