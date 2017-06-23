from __future__ import division

import math
import os

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

from utils import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############# define input options ################
flags = tf.flags

flags.DEFINE_string("output_file", None,
                    "Where the training/test experiment data is stored.")
flags.DEFINE_float("request_ratio", 0.7, "Positive / Negative data ratio. Default value 0.7:0.3")
flags.DEFINE_integer("batch_size", 32, "batch_size (default = 32).")
flags.DEFINE_integer("num_time_steps", 30000, "number of time steps for the AUC optimization")
flags.DEFINE_integer("num_epochs", 10, "number of times to repeat the same experiment")
FLAGS = flags.FLAGS


###################################################

"""
# Process MNIST
"""
# training set
train_num = mnist.train.images.shape[0]
test_num = mnist.test.images.shape[0]
mnist_train = mnist.train.images.reshape([train_num, 28, 28, 1]).astype(np.float32)
train_mean = np.mean(mnist_train, axis=0)
train_norm = np.linalg.norm(mnist_train)
# mean 0
mnist_train = mnist_train - np.stack([train_mean for _ in range(train_num)])
# norm 1
for i in range(train_num):
    mnist_train[i, :] = mnist_train[i, :] / np.linalg.norm(mnist_train[i, :])

# testing set
mnist_test = mnist.test.images.reshape([test_num, 28, 28, 1]).astype(np.float32)
test_mean = np.mean(mnist_test, axis=0)
test_norm = np.linalg.norm(mnist_test)
# mean 0
mnist_test = mnist_test - np.stack([test_mean for _ in range(test_num)])
# norm 1
for i in range(test_num):
    mnist_test[i, :] = mnist_test[i, :] / np.linalg.norm(mnist_test[i, :])

p = FLAGS.request_ratio
# partition training set into +/- groups: ratio=(7:3)
mnist_train_single_labels = []
for i in range(np.shape(mnist.train.labels)[0]):
    if mnist.train.labels[i] > np.ceil(10 * (1 - p)) - 1:
        mnist_train_single_labels.append(1)
    else:
        mnist_train_single_labels.append(0)
# further reshuffle
new_idx = np.random.permutation(mnist.train.labels.shape[0])
mnist_train = mnist_train[new_idx]
mnist_train_single_labels = np.asarray(mnist_train_single_labels)[new_idx]
# partition testing set into +/- groups: ratio=(7:3)
mnist_test_single_labels = []
for i in range(np.shape(mnist.test.labels)[0]):
    if mnist.test.labels[i]>np.ceil(10*(1-p))-1:
        mnist_test_single_labels.append(1)
    else:
        mnist_test_single_labels.append(0)
# as np array
mnist_test_single_labels = np.asarray(mnist_test_single_labels)

W_range = 1.0
batch_size = FLAGS.batch_size


# AUC neural net model
class AUCModel(object):
    global p
    def __init__(self):
        self._build_model()

    def _build_model(self):

        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_sing = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('feature_extraction'):
            #CNN layers:
            self.W_conv0 = tf.get_variable("W_conv0",[5,5,1,4],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1e-3))
            #self.b_conv0 = tf.get_variable("b_conv0",[4],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1e-3))
            self.h_conv0 = conv2d_stride(self.X, self.W_conv0, 2)  # + self.b_conv0
            self.h_relu0 = -tf.nn.elu(self.h_conv0)
            self.bn_conv0 = tf.contrib.layers.batch_norm(self.h_relu0, center=True, scale=True, scope='bn0')

            self.W_conv1 = tf.get_variable("W_conv1", [5, 5, 4, 16], dtype=tf.float32,
                                           initializer=tf.random_normal_initializer(0.0, 1e-3))

            #self.b_conv1 = tf.get_variable("b_conv1",[16],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1e-3))
            self.h_conv1 = conv2d_stride(self.bn_conv0, self.W_conv1, 2)  # + self.b_conv1
            self.h_relu1 = tf.nn.elu(self.h_conv1)
            self.bn_conv1 = tf.contrib.layers.batch_norm(self.h_relu1,
                                                         center=True, scale=True,
                                                         scope='bn1')

            # The feature vector
            self.feature = tf.reshape(self.bn_conv1, [-1, 7*7*16],name='feature')
            self.feature_ave = tf.Variable(tf.zeros([batch_size,7*7*16],
                                           dtype=tf.float32), name='feature_ave')

        with tf.variable_scope('weight'):
            # current copy of w
            self.w = tf.get_variable("w",[7*7*16,1],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1e-3))
            self.w_ave = tf.get_variable("w_ave",[7*7*16,1],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0, 1e-3))
            self.inner_prod = tf.matmul(self.feature, self.w)
            #self.pred = 0.5*tf.sign(self.inner_prod)+0.5
        with tf.variable_scope('network'):
            # current copies of (a,b)
            self.a = tf.Variable(tf.zeros([1], dtype=tf.float32), name='a')
            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32), name='b')
            self.alpha = tf.Variable(tf.zeros([1], dtype=tf.float32), name='alpha')
            # average versions of (a,b)
            self.a_ave = tf.Variable(tf.zeros([1], dtype=tf.float32), name='a_ave')
            self.b_ave = tf.Variable(tf.zeros([1], dtype=tf.float32), name='b_ave')
            self.alpha_ave = tf.Variable(tf.zeros([1], dtype=tf.float32), name='alpha_ave')

            self.loss1 = (1 - p) * tf.multiply(tf.square(self.inner_prod - tf.tile(self.a, [batch_size])), self.y_sing)
            self.loss2 = p * tf.multiply(tf.square(self.inner_prod - tf.tile(self.b, [batch_size])), 1 - self.y_sing)
            self.loss3 = 2 * (1 + self.alpha) * (p*tf.multiply(self.inner_prod, (1 - self.y_sing)) - (1 - p) * tf.multiply(self.inner_prod, self.y_sing)) - p * (1 - p) * tf.square(self.alpha)
            self.loss = tf.reduce_mean(self.loss1 + self.loss2 + self.loss3 + tf.nn.l2_loss(self.inner_prod))


# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = AUCModel()
    learning_rate = tf.placeholder(tf.float32, [])
    weighted_coeff = tf.placeholder(tf.float32, [])
    fraction = tf.divide(learning_rate, weighted_coeff)

    # assign new weighted-averages of (w,a,b,alpha)
    save_w_op = tf.assign(model.w_ave,
                          (1 - fraction) * model.w_ave + fraction * model.w)
    save_a_op = tf.assign(model.a_ave,
                          (1 - fraction) * model.a_ave + fraction * model.a)
    save_b_op = tf.assign(model.b_ave,
                          (1 - fraction) * model.b_ave + fraction * model.b)
    save_alpha_op = tf.assign(model.alpha_ave,
                              (1 - fraction) * model.alpha_ave + fraction * model.alpha)
    reset_a_op = tf.assign(model.a, tf.reshape(0.0, [1]))
    reset_b_op = tf.assign(model.b, tf.reshape(0.0, [1]))
    reset_alpha_op = tf.assign(model.alpha, tf.reshape(0.0, [1]))

    t_vars = tf.trainable_variables()
# -------------------------------------------------------------------------------------------------
#  Optimize (a,b):
    # define min optimizer
    min_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # stochastic descent
    # compute the gradients of a list of vars: a,b
    grads_and_vars_min = min_train_op.compute_gradients(model.loss,[v for v in t_vars if(v.name == 'network/a:0' or v.name == 'network/b:0')])
    min_op = min_train_op.apply_gradients(grads_and_vars_min)
       # clip a and b
    clip_a_op = tf.assign(model.a, tf.clip_by_value(model.a, clip_value_min=-W_range, clip_value_max=W_range))
    clip_b_op = tf.assign(model.b, tf.clip_by_value(model.b, clip_value_min=-W_range, clip_value_max=W_range))

# ------------------------------------------------------------------------------------------------
#  Optimize alpha:
    #  define max optimizer
    max_train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    #  stochastic ascent
    #  compute the gradients of a list of vars: alpha
    grads_and_vars_max=max_train_op.compute_gradients(tf.negative(model.loss),[v for v in t_vars if v.name=='network/alpha:0'])
    max_op = min_train_op.apply_gradients(grads_and_vars_max)
    #  clip alpha
    clip_alpha_op=tf.assign(model.alpha,tf.clip_by_value(model.alpha, clip_value_min=-2*W_range, clip_value_max=2*W_range))

# -------------------------------------------------------------------------------------------------
#  Optimize w:
    #  define weight optimizer
    w_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #  weight optimization:
    grads_and_vars_w = w_train_op.compute_gradients(model.loss, [v for v in t_vars if(v.name == 'weight/w:0')])
    w_op = min_train_op.apply_gradients(grads_and_vars_w)
    #  clip w
    clip_w_op = tf.assign(model.w,tf.clip_by_norm(model.w, clip_norm=W_range, axes=[0]))

# -----------------------------------------------------------------------------------------------
#  Optimize feature:
    #  define feature optimizer
    feat_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #  feature layer optimization:
    feat_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_extraction')
    grads_and_vars_feat = feat_train_op.compute_gradients(model.loss,feat_vars)
    feat_op = feat_train_op.apply_gradients(grads_and_vars_feat)

# ------------------------------------------------------------------------------------------------
# Critic:
    #  apply  SGD/SGA
    critic_op = tf.group(min_op, clip_a_op, clip_b_op, save_a_op, save_b_op,
                         max_op, clip_alpha_op, save_alpha_op,
                         min_op, clip_a_op, clip_b_op, save_a_op, save_b_op,
                         max_op, clip_alpha_op, save_alpha_op,
                         min_op, clip_a_op, clip_b_op, save_a_op, save_b_op,
                         max_op, clip_alpha_op, save_alpha_op,
                         min_op, clip_a_op, clip_b_op, save_a_op, save_b_op,
                         max_op, clip_alpha_op, save_alpha_op)
# -----------------------------------------------------------------------------------------------
# Actor:
    #  apply SGD on feature and weight
    actor_op = tf.group(w_op, clip_w_op, save_w_op, feat_op)
    train_op = tf.group(critic_op, critic_op, critic_op, actor_op)
# ------------------------------------------------------------------------------------------------

# Params
num_steps = FLAGS.num_time_steps


def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        gen_train_batch = batch_generator(
            [mnist_train, mnist_train_single_labels], batch_size)
        gen_test_batch = batch_generator(
            [mnist_test, mnist_test_single_labels], batch_size)
        # Training loop
        wc = 0.0
        # wc_s=0.0
        for i in range(num_steps):
            # Learning rate as in Stochastic Online AUC optimization
            X, y_sing = gen_train_batch.next()

            lr = 2e-5
            wc = wc + lr
            batch_loss, frac, A, B, Alpha, inner_product, _ = sess.run(
                [model.loss, fraction, model.a, model.b, model.alpha, model.inner_prod, train_op],
                feed_dict={model.X: X, model.y_sing: y_sing, learning_rate: lr, weighted_coeff: wc}
            )

            if verbose and i % 200 == 199:
                print '\n\nAUC optimization training, (+/-) ratio =', FLAGS.request_ratio, ':', 1 - FLAGS.request_ratio
                print 'epoch', i, '/', A num_steps
                print 'batch_loss', batch_loss
                print 'learning_rate_s', lr
                print 'fraction_s', frac
                try:
                    batch_auc = metrics.roc_auc_score(y_sing_t, inner_product)
                except Exception:
                    continue

                print 'A', A
                print 'B', B
                print 'Alpha', Alpha
                # print('weighted_coeff',weight)
                print 'sklearn_auc', batch_auc
                # print 'train_acc',accuracy
                print 'inner_product', inner_product.T
                # print 'prediction ', prediction.reshape([batch_size]).astype(int)
                print 'groundtruth', y_sing.astype(int)

        # Compute final evaluation on test data
        mnist_TEST = mnist_test[::50, :]  # only take one-50th of the testing data
        TEST_num = mnist_TEST.shape[0]
        mnist_TEST_single = mnist_test_single_labels[::50]
        inner_product = sess.run(
            [model.inner_prod],
            feed_dict={model.X: mnist_TEST, model.y_sing: mnist_TEST_single}
        )
        inner_product = np.asarray(inner_product).reshape([TEST_num])
        test_auc = metrics.roc_auc_score(mnist_TEST_single, inner_product)
    return test_auc  # , train_auc, train_pre, train_rec


if not FLAGS.output_file:
    raise ValueError("Must set --output_file for experiments")
fout = open('./output/' + FLAGS.output_file, 'a')
fout.write('dataset: MNIST')
fout.write('\noutput_file: ' + FLAGS.output_file)
fout.write('\nbatch_size: ' + str(FLAGS.batch_size))
fout.write('\n(+/-) ratio: ' + str(FLAGS.request_ratio) + ':' + str(1 - FLAGS.request_ratio))
fout.write('\nAUC optimization (method 2) with ' + str(FLAGS.num_time_steps) + ' training steps')
fout.close()
print 'dataset: MNIST'
print 'output_file:', FLAGS.output_file
print '(+/-) ratio:', FLAGS.request_ratio, ' : ', 1 - FLAGS.request_ratio
print '\nauc optimization training'
auc_ave = 0.0
for i in range(FLAGS.num_epochs):
    print 'epoch', i, '/', str(FLAGS.num_epochs)
    fopen = open('./output/' + FLAGS.output_file, 'a')
    fopen.write('\nNumber of experiment ' + str(i) + ' / ' +
                str(FLAGS.num_epochs) + '\n')
    auc = train_and_evaluate('auc', graph, model)
    print 'testing data overall AUC', auc
    fopen.write('testing data overall AUC: ' + str(auc) + '\n')
    auc_ave = (i * auc_ave + auc) / ((i + 1) * 1.0)
    fopen.close()
fopen = open('./output/' + FLAGS.output_file, 'a')
fopen.write('testing data average overall AUC score over ' +
            str(FLAGS.num_epochs) + ' epochs: ' + str(auc_ave) + '\n')
fopen.close()
