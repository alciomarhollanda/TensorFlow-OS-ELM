import numpy
import models
import argparse
import tqdm
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=int, default=784)
parser.add_argument('--units', type=int, default=32)
parser.add_argument('--outputs', type=int, default=784)
parser.add_argument('--loss', default='l1_error')
parser.add_argument('--activation', default='linear')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n', type=int, default=10000)

def main(args):

    os_elm = models.OS_ELM(
        inputs=args.inputs,
        units=args.units,
        outputs=args.outputs,
        loss=args.loss,
        activation=args.activation
    )

    pbar = tqdm.tqdm(total=args.n)
    times = []
    for i in range(args.n):
        x = np.random.normal(size=(args.batch_size, args.inputs))
        y = np.random.normal(size=(args.batch_size, args.outputs))
        stime = time.time()
        os_elm.compute_loss(x, y)
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
