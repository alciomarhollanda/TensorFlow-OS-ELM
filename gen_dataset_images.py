from datasets import get_dataset
import numpy as np
import argparse
import os
from PIL import Image
from datasets import Mnist, Digits, Digits_anomal, Mnist_anomal

parser = argparse.ArgumentParser()
parser.add_argument('output_dir')

def make_dataset_image(x, y, n_classes):
    img = []
    for i in range(n_classes):
        index = (y == i)
        img.append(x[index][0])
    img = np.concatenate(img, axis=-1)
    img = Image.fromarray(img)
    return img

def main(args):
    img_size = (800,80)
    n_classes = 10
    # make result directory(if necessary)
    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)

    # mnist(normal)
    dataset = Mnist()
    (x_mnist, y_mnist), (_, _) = dataset.load_data()
    x_mnist *= 255.
    x_mnist = x_mnist.reshape(-1,28,28).astype(np.uint8)
    y_mnist = np.argmax(y_mnist, axis=-1).astype(np.uint8)
    img = make_dataset_image(x_mnist, y_mnist, n_classes)
    img = img.resize(img_size)
    img.save(os.path.join(args.output_dir, 'mnist.jpg'))

    # mnist(anomal)
    dataset = Mnist_anomal(sigma=0.3)
    (x_mnist, y_mnist), (_, _) = dataset.load_data()
    x_mnist *= 255.
    x_mnist = x_mnist.reshape(-1,28,28).astype(np.uint8)
    y_mnist = np.argmax(y_mnist, axis=-1).astype(np.uint8)
    img = make_dataset_image(x_mnist, y_mnist, n_classes)
    img = img.resize(img_size)
    img.save(os.path.join(args.output_dir, 'mnist_anomal.jpg'))

    # digits(normal)
    dataset = Digits()
    (x_digits, y_digits), (_, _) = dataset.load_data()
    x_digits *= 255.
    x_digits = x_digits.reshape(-1,8,8).astype(np.uint8)
    y_digits = np.argmax(y_digits, axis=-1).astype(np.uint8)
    img = make_dataset_image(x_digits, y_digits, n_classes)
    img = img.resize(img_size)
    img.save(os.path.join(args.output_dir, 'digits.jpg'))

    # digits(anomal)
    dataset = Digits_anomal(sigma=0.3)
    (x_digits, y_digits), (_, _) = dataset.load_data()
    x_digits *= 255.
    x_digits = x_digits.reshape(-1,8,8).astype(np.uint8)
    y_digits = np.argmax(y_digits, axis=-1).astype(np.uint8)
    img = make_dataset_image(x_digits, y_digits, n_classes)
    img = img.resize(img_size)
    img.save(os.path.join(args.output_dir, 'digits_anomal.jpg'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
