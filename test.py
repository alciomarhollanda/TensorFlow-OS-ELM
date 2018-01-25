from keras.models import Sequential
from keras.layers import Dense
from datasets import Digits
import numpy as np
import argparse
import os

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--optimizer',choices=['adam','sgd','adagrad','rmsprop'],default='adam')

def main(args):

    n_classes = 10
    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=(64,)))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    dataset = Digits()
    (x_train,y_train), (x_test,y_test) = dataset.load_data()

    model.fit(
        x=x_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test,y_test),
    )
    # Backpropagation
    # 64 => 256 => 10
    # val_acc: 0.9167(20 epochs, adam),
    # val_acc: 0.8694(20 epochs, sgd),
    # 64 => 256 => 256 => 10
    # val_acc: 0.9111(20 epochs, adam)
    # val_acc: 0.8694(20 epochs, sgd)

    # OS-ELM
    # 64 => 256 => 10
    # val_acc: 0.9379(sigmoid), train_acc:

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
