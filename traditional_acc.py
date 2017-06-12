from __future__ import division

import tensorflow as tf
import numpy as np
from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
"""
# Process MNIST
"""
p=0.7
# training set
mnist_train = mnist.train.images
# kappa: (upper bound) norm of training feature vector
kappa_train = np.linalg.norm(mnist_train)
# testing set
mnist_test  = mnist.test.images
# kappa: (upper bound) norm of testing feature vector
kappa_test = np.linalg.norm(mnist_test)

# partition training set into +/- groups: ratio=(8:2)
mnist_train_bin_labels=[]
mnist_train_single_labels=[]
for i in range(np.shape(mnist.train.labels)[0]):
    if mnist.train.labels[i]>2:
        mnist_train_bin_labels.append([1,0])
        mnist_train_single_labels.append(1)
    else:
        mnist_train_bin_labels.append([0,1])
        mnist_train_single_labels.append(0)
# as np array
new_idx = np.random.permutation(mnist.train.labels.shape[0])
mnist_train = mnist_train[new_idx]
mnist_train_bin_labels=np.asarray(mnist_train_bin_labels)[new_idx]
mnist_train_single_labels=np.asarray(mnist_train_single_labels)[new_idx]

# partition testing set into +/- groups: ratio=(8:2)
mnist_test_bin_labels=[] 
mnist_test_single_labels=[]
for i in range(np.shape(mnist.test.labels)[0]):
    if mnist.test.labels[i]>2:
        mnist_test_bin_labels.append([1,0])
        mnist_test_single_labels.append(1)
    else:
        mnist_test_bin_labels.append([0,1])
        mnist_test_single_labels.append(0)
# as np array
mnist_test_bin_labels=np.asarray(mnist_test_bin_labels)
mnist_test_single_labels=np.asarray(mnist_test_single_labels)

batch_size = 32
# ACC shallow neural net model
class ACCModel(object):
    
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.y_bin = tf.placeholder(tf.float32, [None, 2])
        
        with tf.variable_scope('weight'):
            #self.w 
            self.w = tf.get_variable("w",[784, 2],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,0.01))
            logits = tf.matmul(self.X,self.w)
            self.pred=tf.nn.softmax(logits)
            self.pred_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.y_bin))

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = ACCModel()
    
    learning_rate = tf.placeholder(tf.float32, [])
    
    #define traditional accuracy optimizer 
    regular_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(model.pred_loss)
    
    # Evaluation
    max_pred=tf.cast(tf.argmax(model.pred,1),tf.int64)
    correct_label_pred = tf.equal(tf.argmax(model.y_bin, 1), max_pred)
    label_accuracy = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

# Params
num_steps = 100000

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        
        gen_train_batch = batch_generator(
            [mnist_train, mnist_train_bin_labels], batch_size)
        gen_test_batch = batch_generator(
            [mnist_test, mnist_test_bin_labels], batch_size)
        # Training loop
        for i in range(num_steps):
            
            if training_mode == 'acc':
                lr = 1e-2/np.sqrt(i+1)
                X, y_bin = gen_train_batch.next()
                y_single = y_bin[:,1].T
                _, W, accuracy, batch_pred_loss, pred, prob = sess.run([regular_train_op, model.w, label_accuracy, model.pred_loss, max_pred, model.pred],
                                     feed_dict={model.X: X, model.y_bin: y_bin,learning_rate: lr})
                if verbose and i % 1000 == 0:
                    print '\n\nCross-Entropy Optimization, (+/-) ratio = 7:3'
                    print 'epoch',i
                    AUC=tf.contrib.metrics.streaming_auc(prob, y_bin)
                    sess.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
                    train_AUC=sess.run(AUC)
                    print 'learning_rate',lr
                    print 'batch_pred_loss', batch_pred_loss
                    print 'train_auc',train_AUC
                    #print('W',W)
                    print 'train_acc',accuracy
                    print 'prediction ', pred.reshape([32])
                    print 'groundtruth', y_single.astype(int)

        # Compute final evaluation on test data
        acc = sess.run(label_accuracy,
                            feed_dict={model.X: mnist_test, model.y_bin: mnist_test_bin_labels})
    return acc #, train_auc, train_pre, train_rec


print '\nacc only training'
acc = train_and_evaluate('acc', graph, model)
print 'acc training accuracy:', acc
