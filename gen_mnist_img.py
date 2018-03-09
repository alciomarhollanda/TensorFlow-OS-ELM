from keras.datasets import mnist
from PIL import Image
import numpy as np

(x_train, y_train), (_, _) = mnist.load_data()

imgs = []
for i in range(10):
    ind = (y_train == i)
    x = (x_train[ind])[0]
    imgs.append(x)
imgs = np.concatenate(imgs, axis=-1)
imgs = Image.fromarray(imgs)
imgs.save('images/mnist.bmp')
