import math
from torch.utils.data.dataset import Subset
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def normalize(x, low=0, up=1):
    return (up - low) * (x - x.min()) / (x.max() - x.min()) + low


def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)

def image_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)

def inv_normalize(img, rescale_image=True):
    img = F.normalize(img, mean=[0, 0, 0], std=INV_IMAGENET_STD)
    img = F.normalize(img, mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0])
    return img

def img_to_PIL(img):

    '''
    img: shape [C, H, W]
    '''

    if len(img.shape) != 3:
        raise Exception(f"Mismatch in dimensions of image to show (should be 3), received {img.shape}")

    if type(img) == torch.Tensor:
        img = img.cpu().detach()

    n_ch = img.shape[0]

    img_array = np.transpose(img.numpy(), (1,2,0))
    arr_to_pil = lambda x: Image.fromarray(x.astype('uint8'), 'RGB')

    if n_ch != 3:
        raise Exception(f'Unsupported number of channels ({n_ch}), should be 3')

    norm_img = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    PIL_image = arr_to_pil(norm_img)
    return PIL_image

def get_dataset_fraction(dataset, fraction=0.7):
    n_samples = math.ceil(fraction * len(dataset))
    return Subset(dataset, range(n_samples))