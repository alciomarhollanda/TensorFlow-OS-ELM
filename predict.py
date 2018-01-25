import numpy as np
import os
import argparse
import datasets
import time
import utils
import tqdm
from models import OS_ELM

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--inputs', type=int, default=784)
parser.add_argument('--units', type=int, default=1024)
parser.add_argument('--outputs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--activation', choices=['sigmoid','relu','linear'], default='sigmoid')
parser.add_argument('--loss', choices=['mean_squared_error', 'l1_error'], default='mean_squared_error')
parser.add_argument('--result', default=None)

def main(args):

    # instantiate model
    os_elm = OS_ELM(
        inputs=args.inputs,
        units=args.units,
        outputs=args.outputs,
        activation=args.activation,
        loss=args.loss,
    )

    # prediction
    times = []
    pbar = tqdm.tqdm(total=args.iterations)
    for i in range(args.iterations):
        x = np.random.uniform(size=(args.batch_size, args.inputs))
        y = np.random.uniform(size=(args.batch_size, args.outputs))
        stime = time.time()
        os_elm.compute_loss(x,y)
        etime = time.time()
        times.append(etime - stime)
        pbar.update(n=1)
    pbar.close()
    times = np.array(times)
    print('mean prediction time: %s[sec]' % np.mean(times))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
