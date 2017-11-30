import numpy as np
import models
import datasets
import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('model',choices=['mnist_slp_ae','mnist_cnn_ae'])
parser.add_argument('dataset_normal',choices=['mnist','fashion'])
parser.add_argument('dataset_anomal',choices=['mnist','fashion'])
parser.add_argument('--k',type=float,default=3.)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--batch_size',type=int,default=256)

def add_gaussian_noise(x, mean=0., sigma=0.4):
    gauss = np.random.normal(mean, sigma, size=x.shape)
    x += gauss
    return np.clip(x, 0., 1.)

def main(args):

    dataset_normal = datasets.get_dataset(args.dataset_normal)
    dataset_anomal = datasets.get_dataset(args.dataset_anomal)
    (x_train_normal, _), (x_test_normal, _) = dataset_normal.load_data()
    (_, _), (x_test_anomal, _) = dataset_anomal.load_data()

    # instantiate model
    model = models.get_model(args.model)

    # training
    print('now training phase')
    model.fit(
        x=x_train_normal,
        y=x_train_normal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test_normal,x_test_normal),
        shuffle=True,
        verbose=1)

    # test
    print('now test phase...')
    losses_normal = []
    losses_anomal = []
    for x in x_test_normal:
        x = np.expand_dims(x, axis=0)
        losses_normal.append(model.test_on_batch(x,x))
    for x in x_test_anomal:
        x = np.expand_dims(x, axis=0)
        losses_anomal.append(model.test_on_batch(x,x))
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
