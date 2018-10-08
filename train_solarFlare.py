from sklearn import datasets
from keras.utils import to_categorical
from os_elm import OS_ELM
import numpy as np
import tensorflow as tf
import tqdm
import pandas

def softmax(a):
    c = np.max(a, axis=-1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    return exp_a / sum_exp_a

def main():

    # ===========================================
    # Instantiate os-elm
    # ===========================================
    n_input_nodes = 12
    n_hidden_nodes = 40
    n_output_nodes = 2

    os_elm = OS_ELM(
        # the number of input nodes.
        n_input_nodes=n_input_nodes,
        # the number of hidden nodes.
        n_hidden_nodes=n_hidden_nodes,
        # the number of output nodes.
        n_output_nodes=n_output_nodes,
        # loss function.
        # the default value is 'mean_squared_error'.
        # for the other functions, we support
        # 'mean_absolute_error', 'categorical_crossentropy', and 'binary_crossentropy'.
        loss='mean_squared_error',
        # activation function applied to the hidden nodes.
        # the default value is 'sigmoid'.
        # for the other functions, we support 'linear' and 'tanh'.
        # NOTE: OS-ELM can apply an activation function only to the hidden nodes.
        activation='sigmoid',
    )

    # ===========================================
    # Prepare dataset
    # ===========================================
    n_classes = n_output_nodes

    # load Iris
   # iris = datasets.load_iris()
   # x_iris, t_iris = iris.data, iris.target
    

    # https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
    # load SolarFlare

    url="solarFlare/dataset.zscore.csv"
    dataset = pandas.read_csv(url,sep=";")
    x_solarFlare, t_solarFlare = dataset.values[:, 1:].astype(float) , dataset.values[:,:1].astype(float)


    


    # normalize each column value
    #mean = np.mean(x_solarFlare, axis=0)
    #std = np.std(x_solarFlare, axis=0)
    #x_solarFlare = (x_solarFlare - mean) / std

    # convert label data into one-hot-vector format data.
    t_solarFlare = to_categorical(t_solarFlare, num_classes=n_classes)

    # shuffle dataset
    perm = np.random.permutation(len(x_solarFlare))
    x_solarFlare = x_solarFlare[perm]
    t_solarFlare = t_solarFlare[perm]

    # divide dataset for training and testing
    border = int(len(x_solarFlare) * 0.8)
    x_train, x_test = x_solarFlare[:border], x_solarFlare[border:]
    t_train, t_test = t_solarFlare[:border], t_solarFlare[border:]

    # divide the training dataset into two datasets:
    # (1) for the initial training phase
    # (2) for the sequential training phase
    # NOTE: the number of training samples for the initial training phase
    # must be much greater than the number of the model's hidden nodes.
    # here, we assign int(1.2 * n_hidden_nodes) training samples
    # for the initial training phase.
    border = int(1.2 * n_hidden_nodes)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]
    t_train_init = t_train[:border]
    t_train_seq = t_train[border:]


    # ===========================================
    # Training
    # ===========================================
    # the initial training phase
    pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    pbar.update(n=len(x_train_init))

    # the sequential training phase
    pbar.set_description('sequential training phase')
    batch_size = 8
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i:i+batch_size]
        t_batch = t_train_seq[i:i+batch_size]
        os_elm.seq_train(x_batch, t_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    # ===========================================
    # Prediction
    # ===========================================
    # sample 10 validation samples from x_test
    n = 1464
    x = x_test[:n]
    t = t_test[:n]

    # 'predict' method returns raw values of output nodes.
    y = os_elm.predict(x)
    # apply softmax function to the output values.
    y = softmax(y)

    # check the answers.
    countTrue=0
    countFlare=0
    for i in range(n):
        max_ind = np.argmax(y[i])
        print('========== sample index %d ==========' % i)
        print('estimated answer: class %d' % max_ind)
        print('estimated probability: %.3f' % y[i,max_ind])
        print('true answer: class %d' % np.argmax(t[i]))
        if(max_ind == 1 and max_ind==np.argmax(t[i])):
            countTrue=countTrue+1
        if(1==np.argmax(t[i])):
            countFlare=countFlare+1
    
    print('## true answer estimated true => %d ' % countTrue)
    print('## true answer true => %d ' % countFlare)


    # ===========================================
    # Evaluation
    # ===========================================
    # we currently support 'loss' and 'accuracy' for 'metrics'.
    # NOTE: 'accuracy' is valid only if the model assumes
    # to deal with a classification problem, while 'loss' is always valid.
    # loss = os_elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

    # ===========================================
    # Save model
    # ===========================================
    print('saving model parameters...')
    os_elm.save('./checkpoint/model.ckpt')

    # initialize weights of os_elm
    os_elm.initialize_variables()

    # ===========================================
    # Load model
    # ===========================================
    # If you want to load weights to a model,
    # the architecture of the model must be exactly the same
    # as the one when the weights were saved.
    print('restoring model parameters...')
    os_elm.restore('./checkpoint/model.ckpt')

    # ===========================================
    # ReEvaluation
    # ===========================================
    # loss = os_elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

if __name__ == '__main__':
    main()
