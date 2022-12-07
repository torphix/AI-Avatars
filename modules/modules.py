import torch
import torch.nn as nn
from .unet import Unet
from .utils import angle2matrix
import torch.nn.functional as F
from .resnets import Resnet2d, Resblock2d, Resblock3d


class AppearanceEncoder(nn.Module):
    def __init__(self, image_size=(256,256)):
        super().__init__()
        # 3 Res Block Id Encoder
        self.input_encoder = nn.Sequential(
            # In Layer
            nn.Conv2d(3, 64, (7,7), stride=1, padding=3),
            # Resblocks
            Resblock2d(64, 128, 128, 1, True),
            nn.AdaptiveAvgPool2d((image_size[0]//2, image_size[1]//2)),
            Resblock2d(128, 128, 256, 1, True),
            nn.AdaptiveAvgPool2d((image_size[0]//4, image_size[1]//4)),
            Resblock2d(256, 128, 512, 1, True),
            nn.AdaptiveAvgPool2d((image_size[0]//8, image_size[1]//8)),
            # Output Layer
            nn.Conv2d(512, 2048, 1),
            nn.GroupNorm(8, 2048),
            nn.ReLU(),
        )
        self.volumetric_encoder = nn.Sequential(
            Resblock3d(64, 64, 64, 1, False),
            Resblock3d(64, 64, 64, 1, False),
            Resblock3d(64, 64, 64, 1, False),
        )

        self.global_desc_encoder = Resnet2d(3, 1024, [3,4,6,3], Resblock2d)

    def forward(self, x):
        # Descriptor Encoder
        g_x = self.global_desc_encoder(x)
        # Volumetric Encoder
        x = self.input_encoder(x)
        BS, C, H, W = x.shape
        x = x.reshape(BS, 64, 32, H, W)
        v_x = self.volumetric_encoder(x)
        return v_x, g_x
    

class WarpGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.flow_map_in_src = nn.Conv2d(1024, 512, 1)
        self.flow_map_transform_src = nn.Sequential(
            Resblock3d(512, 256, 256, 1, True),
            nn.Upsample(scale_factor=4),
            Resblock3d(256, 256, 128, 1, True),
            nn.Upsample(scale_factor=2),
            Resblock3d(128, 64, 64, 1, True),
            nn.Upsample(scale_factor=2),
            Resblock3d(64, 32, 32, 1, True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.flow_map_in_drive = nn.Conv2d(1024, 512, 1)
        self.flow_map_transform_drive = nn.Sequential(
            Resblock3d(512, 256, 256, 1, True),
            nn.Upsample(scale_factor=4),
            Resblock3d(256, 256, 128, 1, True),
            nn.Upsample(scale_factor=2),
            Resblock3d(128, 64, 64, 1, True),
            nn.Upsample(scale_factor=2),
            Resblock3d(64, 32, 32, 1, True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.unet3d = Unet(in_d=64, hid_ds=[128, 256, 512], out_d=16, layers_per_block=2)
        self.image_generator = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            Resblock2d(512, 256, 512, 1, False),
            Resblock2d(512, 256, 512, 1, False),
            Resblock2d(512, 256, 512, 1, False),
            Resblock2d(512, 256, 512, 1, False),
            Resblock2d(512, 256, 512, 1, False),
            nn.Upsample(scale_factor=2),
            Resblock2d(512, 256, 512, 1, False),
            nn.Upsample(scale_factor=2),
            Resblock2d(512, 256, 512, 1, False),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, src_features, driver_features, volumetric_features):
        '''
        Inputs: Rotation & Translation, expression features, volumetric features
        Generated from source image
        Source R & T matrix used to center volumetric features
        Drive R & T matrix then used to rotate the features to match driver
        Creates two warp grids for both src and driver: one for position one for emotion
        '''
        # Prepare inputs
        RTs, src_expr = src_features
        RTd, drive_expr = driver_features
        src_input = self.flow_map_in_src(src_expr.unsqueeze(-1).unsqueeze(-1))
        drive_input = self.flow_map_in_drive(drive_expr.unsqueeze(-1).unsqueeze(-1))
        BS, C, H, W = src_input.shape
        src_input = src_input.reshape(BS, C, 1, H, W)
        drive_input = drive_input.reshape(BS, C, 1, H, W)
        # Generate warp flow maps
        src_flow_map = self.flow_map_transform_src(src_input)
        drive_flow_map = self.flow_map_transform_drive(drive_input)

        src_R, src_T = RTs
        drive_R, drive_T = RTd

        # Src Transform (canonicalise)
        src_R = torch.inverse(src_R)
        # Expand for 5D input
        src_R = src_R.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, 32, 32, 32, -1, -1)
        src_T = src_T.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(3).expand(-1, 32, 32, 32, -1, -1)
        src_flow_map = src_flow_map.permute(0,2,3,4,1).unsqueeze(4)
        src_flow_map = (src_flow_map @ src_R) - src_T
        src_flow_map = src_flow_map.squeeze(-2).permute(0,4,1,2,3)        

        warped_features = self.warp_features(volumetric_features, src_flow_map.permute(0,2,3,4,1))
        # Pass features through 3D unet
        x = self.unet3d(warped_features)
        # Drive rotation transform
        drive_R = drive_R.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, 32, 32, 32, -1, -1)
        drive_T = drive_T.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(3).expand(-1, 32, 32, 32, -1, -1)
        drive_flow_map = drive_flow_map.permute(0,2,3,4,1).unsqueeze(4)
        drive_flow_map = (drive_flow_map @ drive_R) - drive_T
        drive_flow_map = drive_flow_map.squeeze(-2).permute(0,4,1,2,3)        

        x = self.warp_features(x, drive_flow_map.permute(0,2,3,4,1))
        BS, C, D, H, W = x.shape
        x = x.reshape(BS, C*D, H, W)
        x = self.image_generator(x)
        return x

    def make_grid(self, N, D, H, W):
        grid_x = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W, 1).expand(N, D, H, -1, -1)
        grid_y = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1, 1).expand(N, D, -1, W, -1)
        grid_z = torch.linspace(-1.0, 1.0, D).view(1, D, 1, 1, 1).expand(N, -1, H, W, -1)
        grid = torch.cat([grid_x, grid_y, grid_z], -1)
        return grid

    def warp_features(self, features, flow_map):
        N, _, D, H, W = features.size() 
        flow_grid = self.make_grid(N, D, H, W).to(features.device)
        # Normalise flow map
        # flow_norm = torch.cat([flow_map[:, :, :, 0:1] / ((W - 1.0) / 2.0), 
                            #    flow_map[:, :, :, 1:2] / ((H - 1.0) / 2.0)], 3)
        # Warp clothing latent space
        warped_features = F.grid_sample(features, flow_grid + flow_map, padding_mode='border', align_corners=True)
        return warped_features

