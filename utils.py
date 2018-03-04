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

def dump_nparray_to_txt(x, path):
    shape = x.shape
    if len(shape) == 1:
        x = x.reshape(1,-1)
    elif len(shape) >= 2:
        temp = np.prod(shape[:-1])
        x = x.reshape(temp,-1)
    else:
        raise ValueError('x is not a numpy array')
    with open(path, 'w') as f:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = x[i,j]
                f.write('%f ' % v)
            f.write('\n')
