import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n_slices',type=int,default=6)
parser.add_argument('--save_fig',default=None)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(args):
    x = np.linspace(-6,6,num=args.n_slices)
    y = sigmoid(x)

    print("[x <= %f]" % (x[0]))
    print("y = 0")
    for i in range(len(x)-1):
        print("[%f < x <= %f]" % (x[i], x[i+1]))
        a = (y[i+1] - y[i]) / (x[i+1] - x[i])
        b = y[i+1] - a * x[i+1]
        print("y = %fx + %f" % (a, b))
    print("[%f < x]" % (x[-1]))
    print("y = 1")

    if args.save_fig:
        plt.plot(x,y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(args.savefig)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
