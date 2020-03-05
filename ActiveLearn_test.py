""""Example usage of BayesianDense layer on MNIST dataset (~1.5% test error). """



import logging
import logging.config
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from tqdm import tqdm
from utils import mnist_reader
from scipy.stats import norm
from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed
# from tensorflow.random import set_seed
import pandas as pd
import os
tf.enable_v2_behavior()
tfd = tfp.distributions
seed(33)
set_random_seed(33)

if not os.path.isdir('results'):
    os.makedirs('results')

def accuracy(model, x, label_true, batch_size):
    """Calculate accuracy of a model"""
    y_pred = model.predict(x, batch_size=batch_size)
    label_pred = np.argmax(y_pred,axis=1)
    correct = np.count_nonzero(label_true == label_pred)
    return 1.0-(float(correct)/float(x.shape[0]))

def one_hot(labels, m):
    """Convert labels to one-hot representations"""
    n = labels.shape[0]
    y = np.zeros((n,m))
    y[np.arange(n),labels.ravel()]=1
    return y

def model(hidden_dim=512, input_dim=28*28, num_train_samples=60000):
    """Create two layer MLP with softmax output"""
    """Creates a Keras model using the LeNet-5 architecture.

     Returns:
         model: Compiled Keras model.
     """
    # KL divergence weighted by the number of training samples, using
    # lambda function to pass as input to the kernel_divergence_fn on
    # flipout layers.
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                              tf.cast(num_train_samples, dtype=tf.float32))

    # Define a LeNet-5 model using three convolutional (with max pooling)
    # and two fully connected dense layers. We use the Flipout
    # Monte Carlo estimator for these layers, which enables lower variance
    # stochastic gradients than naive reparameterization.
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_dim),
        tfp.layers.DenseFlipout(
            hidden_dim, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tfp.layers.DenseFlipout(
            hidden_dim, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tfp.layers.DenseFlipout(
            10, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.softmax)
    ])

    # Model compilation.
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    # We use the categorical_crossentropy loss since the MNIST dataset contains
    # ten labels. The Keras API will then automatically add the
    # Kullback-Leibler divergence (contained on the individual layers of
    # the model), to the cross entropy loss, effectively
    # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model
    return m


def mnist_data():
    """Rescale and reshape MNIST data"""
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    # with gzip.open('mnist.pkl.gz', 'rb') as f:
    #     (x_train, y_train), (x_test, y_test) = pickle.load(f)

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return (x_train, y_train, x_test, y_test)

def fashion_mnist_data():
    """Rescale and reshape MNIST data"""
    datas_path = 'data/fashion-mnist_train.csv'
    df = pd.read_csv(datas_path)
    x_train = df.values[:, 1:]
    y_train = df.values[:, 0]
    datas_path = 'data/fashion-mnist_test.csv'
    df = pd.read_csv(datas_path)
    x_test = df.values[:, 1:]
    y_test = df.values[:, 0]
#     return df.values
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return (x_train, y_train, x_test, y_test)

def trainModel(x_train, y_train, x_test, y_test, m, nb_epoch):

    logging.config.fileConfig('logging.conf')
    path = "output/fmnist"
    if not os.path.exists(path):
        os.makedirs(path)

    nb_epoch = nb_epoch
    batch_size = 128
    k = 10
    decay = 0.96
    lr = 1e-3
    m.summary()
    log = []
    for epoch in range(nb_epoch):
        tr_error = accuracy(m, x_train, y_train, batch_size=batch_size)
        te_error = accuracy(m, x_test, y_test, batch_size=batch_size)
        log.append([tr_error, te_error])
        # m.optimizer.lr.set_value(np.float32(lr))
        logging.info("Epoch: %i/%i, Train: %f, Test: %f, LR: %f"%(epoch, nb_epoch, tr_error, te_error, lr))
        x_train, y_train = shuffle(x_train, y_train)
        callbcks_list = [EarlyStopping(monitor='val_loss',mode='min', patience=10)]
        m.fit(x_train, one_hot(y_train,k), nb_epoch=1, batch_size=batch_size, shuffle=True,
              validation_data=(x_test, one_hot(y_test,k)),callbacks=callbcks_list, verbose=0)
        lr *= decay
        if epoch%10 == 0:
            m.save_weights("%s/checkpoint-%03i.hd5"%(path,epoch))
    m.save_weights('%s/model.hd5'%path)


def first_layer_distribution(first_layer_weights, inp):

    W_mean = first_layer_weights[0]
    W_log_var = first_layer_weights[1]
    bias_mu = first_layer_weights[2]
    # bias_log_var = first_layer_weights[3]
    shape = first_layer_weights[0].shape
#     var = np.zeros(shape[1])
#     mean = np.zeros(shape[1])
    np.reshape(inp, (1, -1))
    mean = np.dot(inp, W_mean)
    var = np.dot(inp**2, np.exp(W_log_var))
    mean = mean + bias_mu
    return mean, var

def remaining_layer_distribution(W, inp_mean, inp_var):
    W_mean = W[0]
    W_log_var = W[1]
    bias_mu = W[2]
    # bias_log_var = W[3]
    shape = W[0].shape
#     var = np.zeros(shape[1])
#     mean = np.zeros(shape[1])
    inp_std = inp_var**0.5

    rectified_mean = inp_mean * norm.cdf(inp_mean/inp_std) + inp_std * norm.pdf(inp_mean/inp_std)

    rectified_2_moment = (inp_mean**2 + inp_var)*norm.cdf(inp_mean/inp_std) + inp_mean*inp_std*norm.pdf(inp_mean/inp_std)

    rectified_var = inp_var**2*(norm.cdf(inp_mean/inp_std)*norm.cdf(-inp_mean/inp_std))+ \
    inp_mean*inp_std*(norm.cdf(-inp_mean/inp_std) - norm.cdf(inp_mean/inp_std))*norm.pdf(inp_mean/inp_std)+\
    inp_var * (norm.cdf(inp_mean/inp_std) - norm.pdf(inp_mean/inp_std)**2 )
    np.reshape(rectified_mean, (1, -1))
    np.reshape(rectified_2_moment, (1, -1))
    np.reshape(rectified_var, (1, -1))
#     print (rectified_mean.shape, rectified_2_moment.shape, rectified_var.shape)
    mean = np.dot(rectified_mean, W_mean)
    var = np.dot(rectified_var, W_mean**2) + np.dot(rectified_2_moment, np.exp(W_log_var))
    mean = mean + bias_mu
    return mean, var


def find_var(weights, inp):
    num_layers = len(weights)//3
    mean, var = first_layer_distribution(weights[0:3], inp)

    for i in range(1, num_layers):
        mean, var = remaining_layer_distribution(weights[3*i:3*i+3], mean, var)
    return mean, var

def ActiveLearnTrain():

    nam = 'active_learn.csv'
    # start by trainig with 1000 samples
    x_train, y_train, x_test, y_test = fashion_mnist_data()
    m=model()
    curr_x_train = x_train[0:1000, :]
    curr_y_train = y_train[0:1000]

    nb_epoch = 50
    trainModel(curr_x_train, curr_y_train, x_test, y_test, m, nb_epoch)

    # add 100 batches at a time to train the rest
    num_rounds = 50
    batch_size = 128
    accuracy_list = [accuracy(m, x_test, y_test, batch_size=batch_size)]
    unexplored = list(range(1000, 60000))
    top_k = 100
    weights = m.get_weights()
    for _ in tqdm(range(num_rounds)):
        min_snr = 10000

        x_unexplored = x_train[unexplored, :]
        print('performing active selection of data')
        mean, var = find_var(weights, x_unexplored)
        pred = np.argmax(mean, axis = 1)
        sinr = np.zeros(len(pred))

        for j in range(len(pred)):
            sinr[j] = (mean[j, pred[j]])**2/(var[j, pred[j]] + sum(mean[j, :]**2) - mean[j, pred[j]]**2)

        best = (sinr).argsort()[:top_k]
        min_ind = [unexplored[i] for i in best]
    #         print (i,j, min_snr)

        curr_x_train = np.vstack((curr_x_train, x_train[min_ind, :]))
        curr_y_train = np.hstack((curr_y_train, y_train[min_ind]))
        for i in min_ind:
            unexplored.remove(i)

        nb_epoch = 50
        trainModel(curr_x_train, curr_y_train, x_test, y_test, m, nb_epoch)
        weights = m.get_weights()
        accuracy_list.append(accuracy(m, x_test, y_test, batch_size=batch_size))
        print(accuracy_list)
    with open('results/'+nam,'w') as fp:
        for a in accuracy_list:
            fp.write(str(a)+',')
        fp.write('\n')


def RandomSample():

    nam = 'random_learn.csv'
    # start by trainig with 1000 samples
    x_train, y_train, x_test, y_test = fashion_mnist_data()
    m=model()
    curr_x_train = x_train[0:1000, :]
    curr_y_train = y_train[0:1000]

    nb_epoch = 100
    trainModel(curr_x_train, curr_y_train, x_test, y_test, m, nb_epoch)

    unexplored = range(1000, 60000)
    num_rounds = 50
    batch_size = 128
    top_k = 100
    accuracy_list = []
    nb_epoch = 50
    for _ in range(num_rounds):

        min_ind = np.random.choice(unexplored, top_k) # modified by vineeth
        curr_x_train = np.vstack((curr_x_train, x_train[min_ind, :]))
        curr_y_train = np.hstack((curr_y_train, y_train[min_ind]))
        min_ind = set(min_ind)
        unexplored = [i for i in unexplored if i not in min_ind]  # modified by vineeth
        trainModel(curr_x_train, curr_y_train, x_test, y_test, m, nb_epoch)
        accuracy_list.append(accuracy(m, x_test, y_test, batch_size=batch_size))
        print(accuracy_list)
    with open('results/'+nam,'w') as fp:
        for a in accuracy_list:
            fp.write(str(a)+',')
        fp.write('\n')


def getResults(pkl_fl):

    with open(pkl_fl) as f:
        res_l = pickle.load(f)
    avg_acc = 1-np.average([i[1] for i in res_l])
    print('Average Test Accuracy:{}'.format(avg_acc))
if __name__ == "__main__":

    # pth = 'output/bayesian_dense/test/'
    # fil = 'log.pkl'
    # getResults(pth+fil)
    ActiveLearnTrain()
    # RandomSample()
