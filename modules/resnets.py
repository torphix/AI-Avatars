import torch
import torch.nn as nn
import torch.nn.functional as F

class Resblock2d(nn.Module):
    '''
    3 x convs
    1 x res_downsample if input shape changes
    output = input + output
    '''
    def __init__(self, in_d, hid_d, out_d, stride, res_downsample):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_d, hid_d, (1,1), 1, 0),
            nn.BatchNorm2d(hid_d))
        self.layer_2 = nn.Sequential(
            nn.Conv2d(hid_d, hid_d, (3,3), stride, 1),
            nn.BatchNorm2d(hid_d))
        self.layer_3 = nn.Sequential(
            nn.Conv2d(hid_d, out_d, (1,1), 1, 0),
            nn.BatchNorm2d(out_d))
        
        if res_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_d, out_d, (1,1), stride),
                nn.BatchNorm2d(out_d))
        else:
            self.downsample = False

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        if self.downsample:
            res = self.downsample(res)
        x = self.relu(x+res)
        return x


class Resnet2d(nn.Module):
    def __init__(self, in_d:int, out_d:int, n_layers:list, ResBlock:nn.Module):
        super().__init__()
        assert len(n_layers) == 4

        self.in_layer = nn.Conv2d(in_d, 64, (7,7), stride=1, padding=3)
        self.block_1 = self.make_blocks(64, 128, n_layers[0], 2, ResBlock)
        self.block_2 = self.make_blocks(128, 128, n_layers[1], 2, ResBlock)
        self.block_3 = self.make_blocks(128, 256, n_layers[2], 2, ResBlock)
        self.block_4 = self.make_blocks(256, 512, n_layers[3], 2, ResBlock)
        self.out_layer = nn.Linear(512, out_d)

    def make_blocks(self, in_d, out_d, n_layers, stride, ResBlock):
        self.layers = nn.ModuleList()
        self.layers.append(ResBlock(in_d, out_d//4, out_d, stride, res_downsample=True))

        for i in range(n_layers-1):
            self.layers.append(
                ResBlock(out_d, out_d//4, out_d, 1, res_downsample=False))

        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = torch.flatten(x, 1)
        x = self.out_layer(x)
        return x


class Resblock3d(nn.Module):
    '''
    3 x convs
    1 x res_downsample if input shape changes
    output = input + output
    '''
    def __init__(self, in_d, hid_d, out_d, stride, res_downsample):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv3d(in_d, hid_d, (1,1,1), 1, 0),
            nn.BatchNorm3d(hid_d))
        self.layer_2 = nn.Sequential(
            nn.Conv3d(hid_d, hid_d, (3,3,3), stride, 1),
            nn.BatchNorm3d(hid_d))
        self.layer_3 = nn.Sequential(
            nn.Conv3d(hid_d, out_d, (1,1,1), 1, 0),
            nn.BatchNorm3d(out_d))
        
        if res_downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_d, out_d, (1,1,1), stride),
                nn.BatchNorm3d(out_d))
        else:
            self.downsample = False

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x.clone()
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))

        if self.downsample:
            res = self.downsample(res)

        x = self.relu(x + res)
        return x




