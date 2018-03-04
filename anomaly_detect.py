import numpy as np
import models
import datasets
import os
import argparse
import tqdm
import time
import utils
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('dataset_normal',choices=['mnist','fashion','digits'])
parser.add_argument('dataset_anomal',choices=['mnist','fashion','digits_anomal','mnist_anomal','fashion_anomal'])
parser.add_argument('--k',type=float,default=3.)
parser.add_argument('--units',type=int,default=1024)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--loss',choices=['mean_squared_error','l1_error'],default='mean_squared_error')
parser.add_argument('--activation',choices=['sigmoid','relu','linear'],default='sigmoid')
parser.add_argument('--dump_path',default=None)

def compute_f_measure(os_elm,x_normal,x_anomal,k):
    losses_normal = []
    losses_anomal = []
    for x in x_normal:
        losses_normal.append(os_elm.compute_loss(x,x))
    for x in x_anomal:
        losses_anomal.append(os_elm.compute_loss(x,x))
    losses = np.concatenate((losses_normal,losses_anomal),axis=0)
    labels = np.concatenate(
        (
            [False for i in range(len(losses_normal))],
            [True for i in range(len(losses_anomal))]
        ),
        axis=0
    )
    mean = np.mean(losses_normal)
    sigma = np.std(losses_normal)
    losses = (losses - mean) / sigma
    thr = k;
    TP = np.sum(labels & (losses > thr))
    precision = TP / np.sum(losses > thr)
    recall = TP / np.sum(labels)
    f_measure = (2. * recall * precision) / (recall + precision)
    return f_measure, precision, recall

def main(args):

    dataset_normal = datasets.get_dataset(args.dataset_normal)
    dataset_anomal = datasets.get_dataset(args.dataset_anomal)
    (x_train_normal, _), (x_test_normal, _) = dataset_normal.load_data()
    (_, _), (x_test_anomal, _) = dataset_anomal.load_data()
    border = int(1.2 * args.units)
    x_train_normal_init = x_train_normal[:border]
    x_train_normal_seq = x_train_normal[border:]
    if args.dump_path:
        utils.dump_nparray_to_txt(x_train_normal_seq, os.path.join(args.dump_path, 'normal_seq_train_data.txt'))
        utils.dump_nparray_to_txt(x_test_normal, os.path.join(args.dump_path, 'normal_test_data.txt'))
        utils.dump_nparray_to_txt(x_test_anomal, os.path.join(args.dump_path, 'abnormal_data.txt'))

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

    # save weights after initial training phase
    if args.dump_path:
        os_elm.save_weights_as_txt(os.path.join(args.dump_path, 'initial_weights.txt'))

    # sequential training
    print('now sequential training phase...')
    pbar = tqdm.tqdm(total=len(x_train_normal_seq))
    step_time_data = []
    for i in range(0,len(x_train_normal_seq),args.batch_size):
        x_batch = x_train_normal_seq[i:i+args.batch_size]
        s_time = time.time()
        os_elm.seq_train(x_batch,x_batch)
        step_time = time.time() - s_time
        step_time_data.append(step_time)
        pbar.update(n=len(x_batch))
    pbar.close()
    if args.dump_path:
        os_elm.save_weights_as_txt(os.path.join(args.dump_path, 'trained_weights.txt'))
        
    print('mean training time: %f [sec/step]' % np.mean(step_time_data))

    # test
    print('now test phase...')
    f_measure, precision, recall = compute_f_measure(os_elm, x_test_normal, x_test_anomal, args.k)
    print('precision: %f' % precision)
    print('recall: %f' % recall)
    print('f_measure: %f' % f_measure)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
