import os
import argparse
import datasets
import models
import numpy as np
import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument('anomal_class',type=int)
parser.add_argument('dataset',choices=['mnist','fashion','digits'])
parser.add_argument('--thr',type=float,default=1.0)
parser.add_argument('--units',type=int,default=1024)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--loss',choices=['mean_squared_error', 'l1_error'],default='mean_squared_error')
parser.add_argument('--activation',choices=['sigmoid','relu'],default='sigmoid')

def main(args):

    anomal_class = args.anomal_class
    dataset = datasets.get_dataset(args.dataset)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # exclude data of anomal class
    mask = (np.argmax(y_train,axis=-1) != anomal_class)
    x_train = x_train[mask]

    # separete dataset
    border = int(args.units * 1.1)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]

    # instantiate model
    os_elm = models.OS_ELM(
        inputs=dataset.inputs,
        units=args.units,
        outputs=dataset.inputs,
        loss=args.loss,
        activation=args.activation)

    # initial training
    print('now initial training phase...')
    os_elm.init_train(x_train_init,x_train_init)

    # sequential training
    print('now sequential training phase...')
    pbar = tqdm.tqdm(total=len(x_train_seq))
    time_data = []
    for i in range(0,len(x_train),args.batch_size):
        x_batch = x_train_seq[i:i+args.batch_size]
        s_time = time.time()
        os_elm.seq_train(x_batch,x_batch)
        e_time = time.time()
        time_data.append(e_time - s_time)
        pbar.update(n=len(x_batch))
    pbar.close()
    print('mean training time: %f [sec/step]' % np.mean(time_data))

    # test
    print('now test phase...')
    for id in range(dataset.num_classes):
        mask = (np.argmax(y_test,axis=-1) == id)
        x_batch = x_test[mask]
        loss = os_elm.compute_loss(x_batch,x_batch)
        print('[%d]: loss = %f' % (id, loss))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
