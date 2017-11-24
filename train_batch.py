import os
import argparse
import datasets
import models

parser = argparse.ArgumentParser()
parser.add_argument(
    'model',
    choices=['mnist_cnn', 'fashion_cnn', 'digits_cnn', 'boston_mlp', 'boston_slp'])
parser.add_argument(
    'dataset',
    choices=['mnist', 'fashion', 'digits', 'boston'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--result', default=None)

def main(args):

    dataset = datasets.get_dataset(args.dataset)
    (x_train,y_train),(x_test,y_test) = dataset.load_data()

    model = models.get_model(args.model, train=True)
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test,y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True)

    if args.result:
        if os.path.exists(args.result) == False:
            os.makedirs(args.result)
        name = '%s_e%d_b%d.h5' % (args.model, args.epochs, args.batch_size)
        model.save(os.path.join(args.result,name))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
