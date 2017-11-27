import numpy as np
import models
import datasets
import os
import argparse
import tqdm
from PIL import Image

def add_gaussian_noise(x, mean=0., sigma=0.4):
    gauss = np.random.normal(mean, sigma, size=x.shape)
    x += gauss
    return np.clip(x, 0., 1.)

def main():

    dataset = datasets.get_dataset('mnist')
    dataset_sub = datasets.get_dataset('fashion')
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    (x_train_sub, y_train_sub), (x_test_sub, y_test_sub) = dataset_sub.load_data()

    # separate x_test into two dataset, normal and anomal
    num_anomal = 1000
    x_test_anomal = x_test_sub[:num_anomal]
    # x_test_anomal = add_gaussian_noise(x_test[:num_anomal])
    x_test_normal = x_test[num_anomal:]

    # separate x_train into two dataset, for seq_train and for init_train
    border = int(1.1 * 1024)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]

    # instantiate model
    os_elm = models.OS_ELM(
        inputs=dataset.inputs,
        units=1024,
        outputs=dataset.inputs,
        loss='mean_squared_error',
        activation='sigmoid')

    # initial training
    print('now initial training phase...')
    os_elm.init_train(x_train_init, x_train_init)

    # sequential training
    print('now sequential training phase...')
    batch_size = 256
    pbar = tqdm.tqdm(total=len(x_train_seq))
    for i in range(0,len(x_train_seq),batch_size):
        x_batch = x_train_seq[i:i+batch_size]
        os_elm.seq_train(x_batch,x_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    # test
    print('now test phase...')
    loss_normal = os_elm.compute_loss(x_test_normal,x_test_normal)
    loss_anomal = os_elm.compute_loss(x_test_anomal,x_test_anomal)
    print('loss_normal: %f' % (loss_normal))
    print('loss_anomal: %f' % (loss_anomal))


if __name__ == '__main__':
    main()
