import numpy
import argparse
import tqdm
import time
import numpy as np
from models import mnist_slp

parser = argparse.ArgumentParser()
parser.add_argument('model',choices=['mnist_slp'])
parser.add_argument('--inputs', type=int, default=784)
parser.add_argument('--units', type=int, default=32)
parser.add_argument('--outputs', type=int, default=784)
parser.add_argument('--loss', default='mean_absolute_error')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n', type=int, default=1000)

def main(args):

    if args.model == 'mnist_slp':
        model = mnist_slp(
            inputs=args.inputs,
            units=args.units,
            outputs=args.outputs
        )

    model.compile(
        optimizer=args.optimizer,
        loss=args.loss
    )

    pbar = tqdm.tqdm(total=args.n)
    times = []
    for i in range(args.n):
        x = np.random.normal(size=(args.batch_size, args.inputs))
        y = np.random.normal(size=(args.batch_size, args.outputs))
        stime = time.time()
        model.test_on_batch(x, y)
        etime = time.time()
        times.append(etime - stime)
        pbar.update(1)
    pbar.close()
    times = np.array(times)
    mean = np.mean(times)
    print('mean prediction time: %f [msec/batch]' % (1000*mean))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
