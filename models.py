import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, mean_squared_error
import numpy as np
import pickle

def my_mean_squared_error(y_true, y_pred):
    return 0.5 * K.mean(K.square(y_pred - y_true), axis=-1)

def mnist_cnn(train=True):

    input = Input(shape=(28*28,))
    x = Reshape((28,28,1))(input)
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input,x)
    if train:
        model.compile(
            optimizer=Adam(),
            loss=categorical_crossentropy,
            metrics=['accuracy'])
    return model

def digits_cnn(train=True):
    input = Input(shape=(8*8,))
    x = Reshape((8,8,1))(input)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input,x)
    if train:
        model.compile(
            optimizer=Adam(),
            loss=categorical_crossentropy,
            metrics=['accuracy'])
    return model

def boston_slp(train=True):
    input = Input(shape=(13,))
    x = Dense(16, activation='relu')(input)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input,x)
    if train:
        model.compile(
            optimizer=Adam(),
            loss=my_mean_squared_error)
    return model

def boston_mlp(train=True):
    input = Input(shape=(13,))
    x = Dense(512, activation='relu')(input)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input,x)
    if train:
        model.compile(
            optimizer=Adam(),
            loss=my_mean_squared_error)
    return model

def get_model(model_name, train=True):
    if model_name == 'mnist_cnn' or model_name == 'fashion_cnn':
        return mnist_cnn(train)
    elif model_name == 'digits_cnn':
        return digits_cnn(train)
    elif model_name == 'boston_mlp':
        return boston_mlp(train)
    elif model_name == 'boston_slp':
        return boston_slp(train)
    else:
        raise Exception('unknown model \'%s\' was spedified.' % model_name)

def load_model(path):
    with open(path, 'rb') as f:
        arc = pickle.load(f)
    os_elm = OS_ELM(
        inputs=arc['inputs'],
        units=arc['units'],
        outputs=arc['outputs'],
        activation=arc['activation'],
        loss=arc['loss'])
    return os_elm

# Network definition
class OS_ELM(object):

    def __mean_squared_error(self, out, y):
        return 0.5 * np.mean((out - y)**2)

    def __l1_error(self, out, y):
        return np.mean(np.abs((out - y)))

    def __accuracy(self, out, y):
        batch_size = len(out)
        accuracy = np.sum((np.argmax(out, axis=1) == np.argmax(y, axis=1)))
        return accuracy / batch_size

    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def __relu(self, x):
        return np.maximum(0, x)

    def __softmax(self, x):
        c = np.max(x, axis=1).reshape(-1, 1)
        upper = np.exp(x - c)
        lower = np.sum(upper, axis=1).reshape(-1, 1)
        return upper / lower

    def __init__(self, inputs, units, outputs, activation='sigmoid', loss='mean_squared_error'):
        self.inputs = inputs
        self.units = units
        self.outputs = outputs
        self.alpha = np.random.rand(inputs, units) * 2.0 - 1.0 # [-1.0, 1.0]
        self.beta = np.random.rand(units, outputs) * 2.0 - 1.0 # [-1.0, 1.0]
        self.bias = np.zeros(shape=(1,self.units))
        self.p = None
        self.activation = activation
        self.loss = loss
        if loss == 'mean_squared_error':
            self.lossfun = self.__mean_squared_error
        elif loss == 'l1_error':
            self.lossfun = self.__l1_error
        else:
            raise Exception('unknown loss function was specified.')
        if activation == 'sigmoid':
            self.actfun = self.__sigmoid
        elif activation == 'relu':
            self.actfun = self.__relu
        else:
            raise Exception('unknown activation function was specified.')

    def __call__(self, x):
        h1 = x.dot(self.alpha) + self.bias
        a1 = self.actfun(h1)
        out = a1.dot(self.beta)
        return out

    def compute_accuracy(self, x, y):
        out = self(x)
        acc = self.__accuracy(out, y)
        return acc

    def compute_loss(self, x, y):
        out = self(x)
        loss = self.lossfun(out,y)
        return loss

    def init_train(self, x, y):
        assert len(x) >= self.units, 'initial dataset size must be >= %d' % (self.units)
        H = self.actfun(x.dot(self.alpha) + self.bias)
        HT = H.T
        self.p = np.linalg.pinv(HT.dot(H))
        self.beta = self.p.dot(HT).dot(y)

    def seq_train(self, x, y):
        H = self.actfun(x.dot(self.alpha))
        HT = H.T
        I = np.eye(len(x))# I.shape = (N, N) N:length of inputa data

        # update P
        temp = np.linalg.pinv(I + H.dot(self.p).dot(HT))    # temp.shape = (N, N)
        self.p = self.p - (self.p.dot(HT).dot(temp).dot(H).dot(self.p))

        # update beta
        self.beta = self.beta + (self.p.dot(HT).dot(y - H.dot(self.beta)))

    def save_weights(self, path):
        weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'p': self.p}
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
            self.alpha = weights['alpha']
            self.beta = weights['beta']
            self.p = weights['p']

    def save(self, path):
        arc = {
            'alpha': self.alpha,
            'beta': self.beta,
            'p': self.p,
            'inputs': self.inputs,
            'units': self.units,
            'outputs': self.outputs,
            'activation': self.activation,
            'loss': self.loss}
        with open(path, 'wb') as f:
            pickle.dump(arc, f)
