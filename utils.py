import numpy as np
import os
import pickle as pkl
from PIL import Image

def min_max_normalize(x, axis=None):
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)
    return (x-min)/(max-min)

def z_normalize(x, axis=None):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean)/std

def resize_array(x, size):
    img = Image.fromarray(x)
    img = img.resize(size=size)
    return np.array(img)
