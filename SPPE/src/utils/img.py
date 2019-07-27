import numpy as np
import cv2
import torch
from PIL import Image
import scipy.misc
from scipy.ndimage import maximum_filter


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def im_to_torch(img):
    img = np.transpose(img,(2,0,1))
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def load_image(img_path):
    return im_to_torch(Image.open(img_path).convert('RGB'))


load_image("/media/ubuntu/文档/code/Pose/img/b.jpg")