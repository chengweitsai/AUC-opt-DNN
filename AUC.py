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
train_num = mnist.train.images.shape[0]
mnist_train_mean = np.mean(mnist.train.images,axis=0)
mnist_train = mnist.train.images-np.stack([mnist_train_mean for _ in range(train_num)])
for i in range(train_num):
    mnist_train[i,:]=mnist_train[i,:]/np.linalg.norm(mnist_train[i,:])
test_num = mnist.test.images.shape[0]
mnist_test_mean = np.mean(mnist.test.images,axis=0)
mnist_test =  mnist.test.images-np.stack([mnist_test_mean for _ in range(test_num)])
for i in range(test_num):
    mnist_test[i,:]=mnist_test[i,:]/np.linalg.norm(mnist_test[i,:])

# partition training set into +/- groups: ratio=(7:3)
mnist_train_single_labels=[]
for i in range(np.shape(mnist.train.labels)[0]):
    if mnist.train.labels[i]>2:
        mnist_train_single_labels.append(1)
    else:
        mnist_train_single_labels.append(0)
# as np array
mnist_train_single_labels=np.asarray(mnist_train_single_labels)

#further reshuffle
new_idx = np.random.permutation(mnist.train.labels.shape[0])
mnist_train = mnist_train[new_idx]
mnist_train_single_labels=np.asarray(mnist_train_single_labels)[new_idx]

# partition testing set into +/- groups: ratio=(7:3)
mnist_test_single_labels=[]
for i in range(np.shape(mnist.test.labels)[0]):
    if mnist.test.labels[i]>2:
        mnist_test_single_labels.append(1)
    else:
        mnist_test_single_labels.append(0)
# as np array
mnist_test_single_labels=np.asarray(mnist_test_single_labels)
W_range=1.5
batch_size = 32
# AUC neural net model
class AUCModel(object):
    global p    
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        
        self.X = tf.placeholder(tf.float64, [None,784])
        self.y_sing = tf.placeholder(tf.float64, [None])
        
        with tf.variable_scope('weight'):
            # current copy of w 
            self.w = tf.Variable(tf.zeros([784,1],dtype=tf.float64), name="w")
            # average version of w
            self.w_ave = tf.Variable(tf.zeros([784,1],dtype=tf.float64), name="w_ave")
            self.inner_prod=tf.matmul(self.X,self.w)
            self.inner_prod_ave=tf.matmul(self.X,self.w_ave)
            self.pred = 0.5*tf.sign(self.inner_prod_ave)+0.5
        with tf.variable_scope('network'):
            # current copies of (a,b)
            self.a = tf.Variable(tf.zeros([1],dtype=tf.float64), name="a")
            self.b = tf.Variable(tf.zeros([1],dtype=tf.float64), name="b")
            self.alpha = tf.Variable(tf.zeros([1],dtype=tf.float64), name="alpha")
            # average versions of (a,b)
            self.a_ave = tf.Variable(tf.zeros([1],dtype=tf.float64), name="a_ave")
            self.b_ave = tf.Variable(tf.zeros([1],dtype=tf.float64), name="b_ave")
            self.alpha_ave = tf.Variable(tf.zeros([1],dtype=tf.float64), name="alpha_ave")
            
            self.loss=tf.reduce_mean((1-p)*tf.multiply(self.y_sing,tf.square(self.inner_prod-tf.tile(self.a,[batch_size])))+p*tf.multiply(1-self.y_sing,tf.square(self.inner_prod-tf.tile(self.b,[batch_size])))+2*(1+self.alpha)*(p*tf.multiply(1-self.y_sing,self.inner_prod)-(1-p)*tf.multiply(self.y_sing,self.inner_prod))-p*(1-p)*tf.square(self.alpha))

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = AUCModel()
    
    learning_rate = tf.placeholder(tf.float64, [])
    weighted_coeff = tf.placeholder(tf.float64, [])
    fraction = tf.divide(learning_rate,weighted_coeff)
    
    # assign new weighted-averages of (w,a,b,alpha)
    save_w_op = tf.assign(model.w_ave, (1-fraction)*model.w_ave+fraction*model.w)
    save_a_op = tf.assign(model.a_ave, (1-fraction)*model.a_ave+fraction*model.a)
    save_b_op = tf.assign(model.b_ave, (1-fraction)*model.b_ave+fraction*model.b)
    save_alpha_op = tf.assign(model.alpha_ave, (1-fraction)*model.alpha_ave+fraction*model.alpha)

    # define min optimizer
    min_train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    
    # define max optimizer
    max_train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    
    # stochastic descent
    t_vars = tf.trainable_variables()
    # compute the gradients of a list of vars: w,a,b
    grads_and_vars_min = min_train_op.compute_gradients(model.loss,[v for v in t_vars if(v.name =='weight/w:0' or v.name =='network/a:0' or v.name == 'network/b:0')])
    min_op = min_train_op.apply_gradients(grads_and_vars_min)
                
    
    clip_a_op=tf.assign(model.a,tf.clip_by_value(model.a, clip_value_min=-W_range, clip_value_max=W_range))
    clip_b_op=tf.assign(model.b,tf.clip_by_value(model.b, clip_value_min=-W_range, clip_value_max=W_range))
    clip_w_op=tf.assign(model.w,tf.clip_by_norm(model.w, clip_norm = W_range,axes=[0])) #axes=[1]: indicates to 'clip_by_norm' for each row of model.w
    
    # stochastic ascent
    # compute the gradients of a list of vars: alpha
    grads_and_vars_max=max_train_op.compute_gradients(tf.negative(model.loss),[v for v in t_vars if v.name=='network/alpha:0'])
    max_op = min_train_op.apply_gradients(grads_and_vars_max)
                
    clip_alpha_op=tf.assign(model.alpha,tf.clip_by_value(model.alpha, clip_value_min=-2*W_range, clip_value_max=2*W_range))
    

    train_op = tf.group(max_op, clip_alpha_op, min_op, clip_a_op, clip_b_op, clip_w_op, 
                        save_w_op, save_a_op, save_b_op, save_alpha_op)
    # Evaluation
    correct_label_pred = tf.equal(tf.cast(model.y_sing,tf.int64), tf.cast(tf.transpose(model.pred),tf.int64))
    label_accuracy = tf.reduce_mean(tf.cast(correct_label_pred, tf.float64))

# Params
num_steps = 200000

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        
        gen_train_batch = batch_generator(
            [mnist_train, mnist_train_single_labels], batch_size)
        gen_test_batch = batch_generator(
            [mnist_test, mnist_test_single_labels], batch_size)
        # Training loop
        wc=0.0
        for i in range(num_steps):
            # Learning rate as in Stochastic Online AUC optimization
            
            # Training step
            if training_mode == 'auc':
                lr = 1e1/np.sqrt(i+1)
                wc = wc + lr
                #lr = 0
                X, y_sing = gen_train_batch.next()

                # fetch loss
                #accuracy, batch_total_loss, prediction, frac, W, A, B, Alpha, inner_product, gvmin, gvmax, correct,  _ = \
                #               sess.run([label_accuracy, model.loss, model.pred, fraction, model.w, model.a, model.b, model.alpha, \
                #                         model.inner_prod, grads_and_vars_min, grads_and_vars_max, correct_label_pred, train_op],
                #               feed_dict={model.X: X, model.y_sing: y_sing, learning_rate: lr,weighted_coeff: wc})
                accuracy, batch_total_loss, prediction, frac, correct,  _ = \
                               sess.run([label_accuracy, model.loss, model.pred, fraction, correct_label_pred, train_op],
                               feed_dict={model.X: X, model.y_sing: y_sing, learning_rate: lr,weighted_coeff: wc})
                if verbose and i % 2000 == 1999:
                    print '\n\nAUC optimization training, (+/-) ratio = 7:3'
                    print 'epoch',i,'/',num_steps
                    AUC=tf.contrib.metrics.streaming_auc(prediction, y_sing)
                    sess.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
                    train_AUC=sess.run(AUC)
                    #gmin,vmin=zip(*gvmin)
                    #gmax,vmax=zip(*gvmax)
                    print 'batch_total_loss',batch_total_loss
                    print 'learning_rate',lr
                    print 'fraction',frac
                    #print 'W',W.T
                    #print 'gmin length',len(gmin)
                    #print 'gmin[0].shape',gmin[0].shape
                    #print 'gradient_w',gmin[0]
                    #print 'gmin[1].shape',gmin[1].shape
                    #print 'gradient_a',gmin[1]
                    #print 'gmin[2].shape',gmin[2].shape
                    #print 'gradient_b',gmin[2]
                    #print 'gvmax length',len(gvmax)
                    #print 'gmax[0].shape',gmax[0].shape
                    #print 'gradient_alpha',gmax[0]
                    #print 'A',A
                    #print 'B',B
                    #print 'Alpha',Alpha
                    #print('weighted_coeff',weight)
                    print 'train_auc',train_AUC
                    print 'train_acc',accuracy
                    #print 'inner_product',inner_product.T
                    print 'prediction ', prediction.reshape([batch_size]).astype(int)
                    print 'groundtruth', y_sing.astype(int)
                    print 'correct    ', correct.reshape([batch_size]).astype(int)
            
        # Compute final evaluation on test data
        acc = sess.run(label_accuracy,
                            feed_dict={model.X: mnist_test, model.y_sing: mnist_test_single_labels})
    return acc #, train_auc, train_pre, train_rec


print '\nauc optimization training'
acc = train_and_evaluate('auc', graph, model)
print 'auc testing accuracy:', acc
