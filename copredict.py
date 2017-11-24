import keras
from keras.models import load_model
import numpy as np
import pickle
import os
import argparse
import datasets
import models

parser = argparse.ArgumentParser()
parser.add_argument('batch_model')
parser.add_argument('speed_model')
parser.add_argument(
    'dataset',
    choices=['mnist','fashion','digits'])
parser.add_argument('--thr',type=float,default=0.1)
parser.add_argument('--batch_size',type=int,default=32)

def softmax(x):
    c = np.max(x, axis=1).reshape(-1, 1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=1).reshape(-1, 1)
    return upper / lower

def compute_score(x):
    x = np.sort(x, axis=-1)
    x = x[:,-1] - x[:,-2]
    return x

def main(args):

    # prepare dataset
    dataset = datasets.get_dataset(args.dataset)
    (_, _), (x_test, y_test) = dataset.load_data()

    # instantiate models
    batch_model = load_model(args.batch_model)
    with open(args.speed_model, 'rb') as f:
        speed_model = pickle.load(f)

    # prediction loop
    for i in range(0,len(x_test),args.batch_size):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        out = os_elm(x_batch)
        prob = softmax(out)
        score = compute_score(prob)
        hit = int(np.sum(score > args.thr))
        print('%d hits in x_test[%d:%d]' % (hit, i, i+args.batch_size))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
