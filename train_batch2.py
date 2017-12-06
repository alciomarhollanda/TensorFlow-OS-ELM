import os
import argparse
import datasets
import models
import tqdm
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    'model',
    choices=['mnist_slp_ae'])
parser.add_argument(
    'dataset',
    choices=['mnist', 'fashion', 'digits', 'boston'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)

def main(args):

    dataset = datasets.get_dataset(args.dataset)
    (x_train,_),(x_test,_) = dataset.load_data()

    model = models.get_model(args.model, train=True)

    n_train = len(x_train)
    for epoch in range(args.epochs):

        perm = np.random.permutation(n_train)
        x_train = x_train[perm]
        time_data = []
        for i in range(0, n_train, args.batch_size):

            x = x_train[i:i+args.batch_size]
            s_time = time.time()
            model.train_on_batch(x,x)
            e_time = time.time()
            time_data.append(e_time - s_time)

        loss = model.test_on_batch(x_test,x_test)
        print('mean training time: %f [sec/step]' % np.mean(time_data))
        print('validation loss: %f' % loss)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
