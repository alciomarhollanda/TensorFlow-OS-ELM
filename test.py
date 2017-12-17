from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--figsize',type=int,default=5)
parser.add_argument('--result',default=os.path.join(curdir,'images'))
def main(args):

    # Make result directory
    if os.path.exists(args.result) == False:
        os.makedirs(args.result)

    # Load the digits dataset
    digits = datasets.load_digits()

    # normal
    x_normal = np.concatenate(digits.images[:10],axis=-1)
    plt.figure(1, figsize=(args.figsize,args.figsize))
    plt.imshow(x_normal, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig(os.path.join(args.result,'digits_normal.png'))
    plt.clf()

    # inv
    x_inv = 16.0 - x_normal
    plt.figure(1, figsize=(args.figsize,args.figsize))
    plt.imshow(x_inv, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig(os.path.join(args.result,'digits_inv.png'))
    plt.clf()

    # noise
    gauss = np.random.normal(loc=0., scale=16.0*0.3, size=x_normal.shape)
    x_noise = np.clip(x_normal+gauss,a_min=0.,a_max=16.0)
    plt.figure(1, figsize=(args.figsize,args.figsize))
    plt.imshow(x_noise, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.savefig(os.path.join(args.result,'digits_noise.png'))
    plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
