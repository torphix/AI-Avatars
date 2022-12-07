import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, 
                in_d,
                hid_ds,
                out_d,
                layers_per_block=2,
                upsample_type='transpose',
                use_condition=False):
        super().__init__()
        '''Number of Unet layers is == # of hid_ds'''
        print(f'Using condition for Unet: {use_condition}')
        self.use_condition = use_condition
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        self.in_layer = nn.Sequential(
            nn.Conv3d(in_d, hid_ds[0], 3, 1, 1),
            nn.BatchNorm3d(hid_ds[0]),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        for i in range(len(hid_ds)-1):
            self.downsample_layers.append(
                UnetDownsample(hid_ds[i],
                               hid_ds[i+1],
                               layers_per_block))

        self.mid_layer = nn.Sequential(
            nn.Conv3d(
                hid_ds[-1]*2 if use_condition else hid_ds[-1],
                hid_ds[-1],
                3, 1, 1),
            nn.BatchNorm3d(hid_ds[-1]),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        
        hid_ds.reverse()
        for i in range(len(hid_ds)-1):
            self.upsample_layers.append(
                UnetUpsample(
                    hid_ds[i]*2,
                    hid_ds[i+1],
                    layers_per_block,
                    type=upsample_type))

        self.out_layer = nn.Sequential(
            nn.Conv3d(hid_ds[-1], out_d, 1),
            nn.Tanh()
        )
        
        
    def forward(self, x, condition=None):
        img_zs = []
        x = x.float()

        x = self.in_layer(x)
        # Down
        for i, layer in enumerate(self.downsample_layers):
            x = layer(x)
            img_zs.append(x)
        img_zs.reverse()

        if self.use_condition:
            condition = condition.unsqueeze(-1).unsqueeze(-1)
            condition = condition.expand(
                x.shape[0], -1, x.shape[-1], x.shape[-1])
            x = torch.cat((x, condition), dim=1)
        x = self.mid_layer(x)
        
        # Up
        for i, layer in enumerate(self.upsample_layers):
            x = torch.cat((x, img_zs[i]), dim=1)
            x = layer(x) 
        return self.out_layer(x)

    
class UnetDownsample(nn.Module):
    def __init__(self, in_d, out_d, n_layers=2):
        super().__init__()
        '''
        Given input size, output size & stride 
        kernel size is automatically selected
        '''
        self.layers = nn.ModuleList()

        # Conv Res block
        for i in range(n_layers):
            self.layers.append(
                    ResDoubleConv(in_d))

        # Down Conv
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(in_d, out_d, 3, 2, 1),
                nn.BatchNorm3d(out_d),
                nn.ReLU()
            ))

        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
        

class UnetUpsample(nn.Module):
    def __init__(self, in_d, out_d, n_layers=2, type='transpose'):
        super().__init__()
        '''
        Given input size, output size & stride 
        kernel size is automatically selected
        '''
        assert type in ['transpose', 'upsample'], \
            'UnetUpsample layers supports transpose or upsample only'

        self.layers = nn.ModuleList()

        if type == 'transpose':        
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_d, out_d, kernel_size=4,
                                        padding=1, stride=2),
                    nn.BatchNorm3d(out_d),
                    nn.ReLU(),
                    nn.Dropout3d(0.3),
                    ResDoubleConv(out_d)
            ))

        elif type == 'upsample':
            self.layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(in_d, out_d, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(out_d),
                    nn.ReLU(),
                    nn.Dropout(0.3)
            ))
        
        for i in range(n_layers):
            self.layers.append(
                ResDoubleConv(out_d))
                
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
        

class ResDoubleConv(nn.Module):
    def __init__(self, in_d, kernel=3, stride=1, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_d, in_d, kernel, stride, padding),
            nn.BatchNorm3d(in_d),
            nn.ReLU(),
            nn.Conv3d(in_d, in_d, kernel, stride, padding),
            nn.BatchNorm3d(in_d),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        res = x
        out = self.layers(x) + res
        out = self.relu(out)
        return out