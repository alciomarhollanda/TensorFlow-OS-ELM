import numpy as np
import models
import datasets
import os
import argparse
import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('dataset_normal',choices=['mnist','fashion'])
parser.add_argument('dataset_anomal',choices=['mnist','fashion'])
parser.add_argument('--k',type=float,default=3.)
parser.add_argument('--units',type=int,default=1024)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--loss',choices=['mean_squared_error','l1_error'],default='mean_squared_error')
parser.add_argument('--activation',choices=['sigmoid','relu','linear'],default='sigmoid')

def add_gaussian_noise(x, mean=0., sigma=0.4):
    gauss = np.random.normal(mean, sigma, size=x.shape)
    x += gauss
    return np.clip(x, 0., 1.)

def main(args):

    dataset_normal = datasets.get_dataset(args.dataset_normal)
    dataset_anomal = datasets.get_dataset(args.dataset_anomal)
    (x_train_normal, _), (x_test_normal, _) = dataset_normal.load_data()
    (_, _), (x_test_anomal, _) = dataset_anomal.load_data()
    border = int(1.1 * args.units)
    x_train_normal_init = x_train_normal[:border]
    x_train_normal_seq = x_train_normal[border:]

    # instantiate model
    os_elm = models.OS_ELM(
        inputs=dataset_normal.inputs,
        units=args.units,
        outputs=dataset_normal.inputs,
        loss=args.loss,
        activation=args.activation)

    # initial training
    print('now initial training phase...')
    os_elm.init_train(x_train_normal_init, x_train_normal_init)

    # sequential training
    print('now sequential training phase...')
    pbar = tqdm.tqdm(total=len(x_train_normal_seq))
    for i in range(0,len(x_train_normal_seq),args.batch_size):
        x_batch = x_train_normal_seq[i:i+args.batch_size]
        os_elm.seq_train(x_batch,x_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    # test
    print('now test phase...')
    losses_normal = []
    losses_anomal = []
    for x in x_test_normal:
        losses_normal.append(os_elm.compute_loss(x,x))
    for x in x_test_anomal:
        losses_anomal.append(os_elm.compute_loss(x,x))
    loss_normal = np.mean(losses_normal)
    loss_anomal = np.mean(losses_anomal)
    print('loss_normal: %f' % (loss_normal))
    print('loss_anomal: %f' % (loss_anomal))

    losses = np.concatenate((losses_normal,losses_anomal),axis=0)
    labels = np.concatenate(
        (
            [False for i in range(len(losses_normal))],
            [True for i in range(len(losses_anomal))]
        ),
        axis=0
    )
    losses = (losses - loss_normal) / np.std(losses_normal)
    thr = args.k;
    TP = np.sum(labels & (losses > thr))
    precision = TP / np.sum(losses > thr)
    recall = TP / np.sum(labels)
    f_measure = (2. * recall * precision) / (recall + precision)
    print('precision: %f' % precision)
    print('recall: %f' % recall)
    print('f_measure: %f' % f_measure)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
