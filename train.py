import numpy as np
import os
import argparse
import models
import datasets
import time
import utils
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    choices=['mnist', 'fashion', 'digits', 'boston'],
    default='mnist')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--units', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--activation', choices=['sigmoid','relu'], default='sigmoid')
parser.add_argument('--loss', choices=['mean_squared_error'], default='mean_squared_error')
parser.add_argument('--result', default=None)

def main(args):

    # prepare dataset
    dataset = datasets.get_dataset(args.dataset)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    border = int(args.units * 1.1)
    x_train_init, x_train_seq = x_train[:border], x_train[border:]
    y_train_init, y_train_seq = y_train[:border], y_train[border:]

    test_loss = []
    test_acc = []
    seq_time = []
    init_time = []
    pred_time = []
    for epoch in range(args.epochs):

        # instantiate model
        os_elm = models.OS_ELM(
            inputs=dataset.inputs,
            units=args.units,
            outputs=dataset.outputs,
            activation=args.activation,
            loss=args.loss)

        # initial training
        stime = time.time()
        os_elm.init_train(x_train_init, y_train_init)
        init_time.append(time.time() - stime)

        # sequential training
        for i in range(0, len(x_train_seq), args.batch_size):
            x_batch = x_train_seq[i:i+args.batch_size]
            y_batch = y_train_seq[i:i+args.batch_size]
            stime = time.time()
            os_elm.seq_train(x_batch, y_batch)
            seq_time.append(time.time() - stime)

        # evaluation
        # prediction time
        stime = time.time()
        os_elm(np.zeros(shape=(args.batch_size,dataset.inputs)))
        pred_time.append(time.time() - stime)

        # loss and accuracy
        test_loss.append(os_elm.compute_loss(x_test,y_test))
        if dataset.type == 'classification':
            test_acc.append(os_elm.compute_accuracy(x_test,y_test))

    # show result
    print('********** |dataset:%s|units:%d|batch_size:%d| **********' % (
        args.dataset, args.units, args.batch_size
    ))
    print('test_loss: %f' % np.mean(test_loss))
    if dataset.type == 'classification':
        print('test_acc: %f' % np.mean(test_acc))
    print('init_time: %f[sec]' % np.mean(init_time))
    print('seq_time: %f[sec/batch]' % np.mean(seq_time))
    print('pred_time: %f[sec/batch]' % np.mean(pred_time))

    # save model
    if args.result:
        if os.path.exists(args.result) == False:
            os.makedirs(args.result)
        fname = '%s_u%d_b%d.pkl' % (args.dataset, args.units, args.batch_size)
        with open(os.path.join(args.result,fname), 'wb') as f:
            pickle.dump(os_elm, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
