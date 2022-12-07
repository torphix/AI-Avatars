import torch
import numpy as np
from PIL import Image
# from modules.inference import Inference
from torchvision.utils import save_image
from modules.utils import draw_projection

def test_headpose(model_ckpt:str, sample_image:str):
    if sample_image.endswith('pt'):
        sample_image = torch.load(sample_image)
    else:
        sample_image = Image.load(sample_image)
        sample_image = torch.from_numpy(np.array(sample_image))

    inference = Inference(model_ckpt)
    R, T = inference.headpose_inference(sample_image)
    sample_image = np.array(sample_image)
    draw_projection(sample_image, R, T.numpy(), landmarks, color=(224, 255, 255))
    im = Image.fromarray(sample_image)
    im.save('out.png')


