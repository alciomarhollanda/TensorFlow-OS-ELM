import numpy as np
import os
import argparse
import datasets
import models

parser = argparse.ArgumentParser()
parser.add_argument('model',choices=[
    'mnist_cnn_ae',
    'mnist_slp_ae',
    'fashion_cnn_ae',
    'fashion_slp_ae'])
parser.add_argument('anomal_class',type=int)
parser.add_argument('dataset',choices=['mnist','fashion','digits'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--result', default=None)

def main(args):

    anomal_class = args.anomal_class
    dataset = datasets.get_dataset(args.dataset)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # exclude data of anomal class
    mask = (np.argmax(y_train,axis=-1) != anomal_class)
    x_train = x_train[mask]

    # instantiate model
    model = models.get_model(args.model)

    # training
    print('now training phase...')
    model.fit(
        x_train,
        x_train,
        epochs=args.epochs,
        batch_size=args.batch_size)

    # test
    print('now test phase...')
    for id in range(dataset.num_classes):
        mask = (np.argmax(y_test,axis=-1) == id)
        x_batch = x_test[mask]
        loss = model.test_on_batch(x_batch,x_batch)
        print('[%d]: loss = %f' % (id, loss))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
