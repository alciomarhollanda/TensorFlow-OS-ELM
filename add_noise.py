import os
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input_img')
parser.add_argument('output_img')

def main(args):
    input_img = Image.open(args.input_img)
    img_data = np.array(input_img)
    img_data = img_data.astype(np.float32) / 255.
    gauss = np.random.normal(loc=0., scale=0.4, size=img_data.shape)
    img_data = np.clip(img_data+gauss, 0., 1.)
    img_data = (img_data * 255.).astype(np.uint8)
    output_img = Image.fromarray(img_data)
    output_img.save(args.output_img)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
