import os
import math
import torch
import shutil
import hashlib
import tempfile
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from urllib.request import urlopen, Request

class LossModule():
    def __init__(self):
        self.regular_vgg_loss = VGGLoss()
        self.face_vgg_loss = FaceLoss((256,256))

    def generator_loss(self, src, drive, generated,
                       patch_fake_pred, patch_realred,
                       scale_fake_pred, scale_realred,
                       patch_fake_latents, patch_real_latents,
                       scale_fake_latents, scale_real_latents):

        reg_vgg_loss = self.regular_vgg_loss(generated, drive)
        face_vgg_loss = self.face_vgg_loss(generated, drive)

        # Adverserial loss
        scale_loss, patch_loss = 0, 0
        for i in range(len(scale_fake_pred)):
            scale_loss += torch.mean(1-scale_fake_pred[i].float())**2
        for i in range(len(patch_fake_pred)):
            patch_loss += torch.mean(1-patch_fake_pred[i].float())**2

        scale_latent_loss, patch_latent_loss = 0, 0
        for i in range(len(scale_fake_latents)):
            for j in range(len(scale_fake_latents[i])):
                scale_latent_loss += torch.mean(torch.abs(
                    scale_real_latents[i][j] - scale_fake_latents[i][j]))
        for i in range(len(patch_fake_latents)):
            for j in range(len(patch_fake_latents[i])):
                patch_latent_loss += torch.mean(torch.abs(
                    patch_real_latents[i][j] - patch_fake_latents[i][j]))
        return {
            'total_loss': reg_vgg_loss + face_vgg_loss + scale_loss + patch_loss + scale_latent_loss + patch_latent_loss,
            'reg_vgg_loss': reg_vgg_loss,
            'face_vgg_loss': face_vgg_loss,
            'scale_loss': scale_loss,
            'patch_loss': patch_loss,
            'scale_latent_loss': scale_latent_loss,
            'patch_latent_loss': patch_latent_loss,
        }

    def discriminator_loss(self, generated, real):
        '''
        Values should be a list of intermediate resolution
        GANS as multiscale patch GAN is used
        '''
        loss = 0
        for i in range(len(generated)):
            # MSE
            real_loss = torch.mean((1-real[i])**2)
            gen_loss = torch.mean((generated[i]**2))
            loss += (real_loss + gen_loss)
        return loss


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        Gx = gram_matrix(x)
        Gy = gram_matrix(y)
        return F.mse_loss(Gx, Gy) * 30000000


class VGGLoss(nn.Module):
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        # self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.weights = [5.0, 1.0, 0.5, 0.4, 0.8]
        # self.style_weights = [10e4, 1000, 50, 15, 50]

    def forward(self, x, y, style=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = (self.weights[i] *
                             self.criterion(x_vgg[i], y_vgg[i].detach()))
                this_style_loss = (self.style_weights[i] *
                                   self.style_criterion(x_vgg[i], y_vgg[i].detach()))
                loss += this_loss
                style_loss += this_style_loss
            return loss, style_loss

        for i in range(len(x_vgg)):
            this_loss = (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
            loss += this_loss
        return loss


class FaceLoss(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.mtcnn = MTCNN(image_size=img_size, margin=0)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()

    def forward(self, src_img, tgt_img):
        with torch.no_grad():
            src_img = self.mtcnn(src_img)
            tgt_img = self.mtcnn(tgt_img)
            _, src_features = self.resnet(src_img)
            _, tgt_features = self.resnet(tgt_img)

        loss, style_loss = 0,0
        for i in range(len(src_features)):
            style_loss += self.style_criterion(src_features[i], tgt_features[i])
            loss += self.criterion(src_features[i], tgt_features[i])
        return style_loss, loss


class InceptionResnetV1(nn.Module):
    """
    https://github.com/timesler/facenet-pytorch
    Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            self.load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        outputs = []
        x = self.conv2d_1a(x)
        outputs += x
        x = self.conv2d_2a(x)
        outputs += x
        x = self.conv2d_2b(x)
        outputs += x
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        outputs += x
        x = self.repeat_1(x)
        outputs += x
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        outputs += x
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        outputs += x
        x = self.block8(x)
        outputs += x
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x, outputs

    def load_weights(self, mdl, name):
        """Download pretrained state_dict and load into model.

        Arguments:
            mdl {torch.nn.Module} -- Pytorch model.
            name {str} -- Name of dataset that was used to generate pretrained state_dict.

        Raises:
            ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
        """
        if name == 'vggface2':
            path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
        elif name == 'casia-webface':
            path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
        else:
            raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

        model_dir = 'model_weights'
        os.makedirs(model_dir, exist_ok=True)

        cached_file = os.path.join(model_dir, os.path.basename(path))
        if not os.path.exists(cached_file):
            self.download_url_to_file(path, cached_file)

        state_dict = torch.load(cached_file)
        mdl.load_state_dict(state_dict)

        
    def download_url_to_file(self, url, dst, hash_prefix=None, progress=True):
        r"""Download object at the given URL to a local path.
        Args:
            url (string): URL of the object to download
            dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
            hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
                Default: None
            progress (bool, optional): whether or not to display a progress bar to stderr
                Default: True
        Example:
            >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
        """
        file_size = None
        # We use a different API for python2 since urllib(2) doesn't recognize the CA
        # certificates in older Python
        req = Request(url, headers={"User-Agent": "torch.hub"})
        u = urlopen(req)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

        # We deliberately save it in a temp file and move it after
        # download is complete. This prevents a local working checkpoint
        # being overridden by a broken download.
        dst = os.path.expanduser(dst)
        dst_dir = os.path.dirname(dst)
        f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

        try:
            if hash_prefix is not None:
                sha256 = hashlib.sha256()
            with tqdm(total=file_size, disable=not progress,
                    unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if hash_prefix is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

            f.close()
            if hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[:len(hash_prefix)] != hash_prefix:
                    raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                    .format(hash_prefix, digest))
            shutil.move(f.name, dst)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)





class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out