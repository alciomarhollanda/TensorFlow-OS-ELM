import os
import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense
import tqdm
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--inputs', type=int, default=784)
parser.add_argument('--units', type=int, default=1024)
parser.add_argument('--outputs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--result', default=None)

def main(args):

    # instantiate model
    model = Sequential()
    model.add(Dense(args.units, activation='relu', input_shape=(args.inputs,)))
    model.add(Dense(args.outputs, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # prediction
    times = []
    pbar = tqdm.tqdm(total=args.iterations)
    for i in range(args.iterations):
        x = np.random.uniform(size=(args.batch_size, args.inputs))
        y = np.random.uniform(size=(args.batch_size, args.outputs))
        stime = time.time()
        model.test_on_batch(x,y)
        etime = time.time()
        times.append(etime - stime)
        pbar.update(n=1)
    pbar.close()
    times = np.array(times)
    print('mean prediction time: %f[sec]' % np.mean(times))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
