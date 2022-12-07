import torch
import itertools
from .utils import open_configs
import pytorch_lightning as ptl
import torch.nn.functional as F
from modules.model import BaseModel
from modules.losses import LossModule
from torch.utils.data import DataLoader
from headpose.headpose import HeadposeInference
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.datasets import HeadposeDataset, BaseDataset, split_dataset
from modules.discriminators import MultiPatchDiscriminator, MultiScaleDiscriminator

class BaseTrainModule(ptl.LightningModule):
    def __init__(self, train_config, data_config):
        super().__init__()
        self.train_config = train_config
        self.generator = BaseModel()
        self.headpose = HeadposeInference()
        self.scale_D = MultiScaleDiscriminator(3)
        self.patch_D = MultiPatchDiscriminator(3)

        self.loss = LossModule()

        self.dataset = BaseDataset(data_config['data_path'], 
                                   f"{data_config['data_path']}/data.txt")

        self.train_ds, self.val_ds = split_dataset(
            self.dataset, train_config['split_size'])

    def forward(self, batch, batch_idx, optimizer_idx):
        g_out = self.generator(batch['src_1'], batch['drive_1'])

        # Train discriminator
        if optimizer_idx == 0:
            losses = self.discriminator_step(g_out, batch, batch_idx)

        # Train generator
        elif optimizer_idx == 1:
            losses = self.generator_step(g_out, batch, batch_idx)

        return losses['total_loss']

    def generator_step(self, generated, src, drive, batch_idx):
        patch_fake_pred, patch_fake_latents = self.patch_D(generated)
        patch_real_pred, patch_real_latents = self.patch_D(drive)
        scale_fake_pred, scale_fake_latents = self.scale_D(generated)
        scale_real_pred, scale_real_latents = self.scale_D(drive)
        # Losses
        losses = self.loss.generator_loss(
            src, drive, generated,
            patch_fake_pred, patch_real_pred,
            scale_fake_pred, scale_real_pred,
            patch_fake_latents, patch_real_latents,
            scale_fake_latents, scale_real_latents,
        )
        return losses

    def discriminator_step(self, generated, src, drive, batch_idx):
        # Scale
        scale_fake_pred, _ = self.scale_D(generated)
        scale_real_pred, _ = self.scale_D(drive)
        # Patch 
        patch_fake_pred, _ = self.patch_D(generated)
        patch_real_pred, _ = self.patch_D(drive)
        # Loss
        scale_loss = self.loss.discriminator_loss(scale_fake_pred, scale_real_pred)
        patch_loss = self.loss.discriminator_loss(patch_fake_pred, patch_real_pred)
        return {
            'total_loss':scale_loss + patch_loss,
            'scale_loss':scale_loss,
            'patch_loss':patch_loss,
        }

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, True)
        self.log_values({'train_loss':loss})
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, False)
        self.log_values({'val_loss':loss})
        return {'loss':loss}

    def configure_optimizers(self):
        D_parameters = itertools.chain(self.patch_D.parameters(), self.scale_D.parameters())
        opt_d = torch.optim.Adam(D_parameters, 
                                 **self.train_config['d_optim'])
        opt_g = torch.optim.Adam(self.generator.parameters(), 
                                 **self.train_config['g_optim'])
        scheduler_g = {"scheduler": ReduceLROnPlateau(opt_g, **self.train_config['g_scheduler']), 'moniter':'val_loss'}
        scheduler_d = {"scheduler": ReduceLROnPlateau(opt_d, **self.train_config['d_scheduler']), 'moniter':'val_loss'}
        return [opt_d, opt_g]#, [scheduler_d, scheduler_g]   

    def log_values(self, scalars):
        logger = self.logger.experiment
        for k, v in scalars.items():
            logger.add_scalar(k,v)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          **self.train_config['train_dl'],
                          collate_fn=self.dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                        batch_size=8, 
                        shuffle=False, 
                        collate_fn=self.dataset.collate_fn)


class HeadposeTrainModule(ptl.LightningModule):
    def __init__(self, train_config, data_config):
        super().__init__()
        self.train_config = train_config
        self.model = HeadposePredictor()
        self.dataset = HeadposeDataset(data_config['data_path'], 
                                       f"{data_config['data_path']}/data.txt")

        self.train_ds, self.val_ds = split_dataset(
            self.dataset, train_config['split_size'])

    def forward(self, batch, train_step):
        # Use log values of displacements as training target
        outputs = self.model(batch['frames'].permute(0, 3, 1, 2).contiguous())
        loss = F.mse_loss(outputs, batch['headposes'].float())
        return loss 

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, True)
        self.log_values({'train_loss':loss})
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, False)
        self.log_values({'val_loss':loss})
        return {'loss':loss}

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.model.parameters(), 
                                 **self.train_config['optimizer'])
        self.plateau_scheduler = ReduceLROnPlateau(self.opt, **self.train_config['scheduler'])
        return self.opt

    def log_values(self, scalars):
        logger = self.logger.experiment
        for k, v in scalars.items():
            logger.add_scalar(k,v)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          **self.train_config['train_dl'],
                          collate_fn=self.dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                        batch_size=8, 
                        shuffle=False, 
                        collate_fn=self.dataset.collate_fn)

def train(train_model):
    train_config, data_config, model_config = open_configs(['train', 'data', 'model'])
    # Load Model
    if train_model == 'headpose':
        train_module = HeadposeTrainModule(train_config['headpose'], data_config)
    elif train_model == 'base':
        train_module = BaseTrainModule(train_config['base'], data_config)

    trainer = ptl.Trainer(**train_config['trainer'])
    trainer.fit(train_module)
    torch.save(train_module.state_dict(), f'model_weights/{train_model}.pth')

