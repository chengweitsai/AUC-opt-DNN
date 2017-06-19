from __future__ import division
import os
import tensorflow as tf
import numpy as np
from utils import *
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
# Process MNIST
"""
# training set
train_num = mnist.train.images.shape[0]
test_num = mnist.test.images.shape[0]
mnist_train = mnist.train.images.reshape([train_num,28,28,1]).astype(np.float32)

train_mean = np.mean(mnist_train,axis=0)
train_norm = np.linalg.norm(mnist_train)
# mean 0
mnist_train = mnist_train - np.stack([train_mean for _ in range(train_num)])
# normalization
mnist_train=mnist_train/255

# testing set
mnist_test  = mnist.test.images.reshape([test_num,28,28,1]).astype(np.float32)

test_mean = np.mean(mnist_test,axis=0)
test_norm = np.linalg.norm(mnist_test)
# mean 0
mnist_test = mnist_test - np.stack([test_mean for _ in range(test_num)])
# normalization
mnist_test=mnist_test/255

p=0.5
# partition training set into +/- groups: ratio=(7:3)
mnist_train_single_labels=[]
for i in range(np.shape(mnist.train.labels)[0]):
    if mnist.train.labels[i]>4:
        mnist_train_single_labels.append(1)
    else:
        mnist_train_single_labels.append(0)
# further reshuffle
new_idx = np.random.permutation(mnist.train.labels.shape[0])
mnist_train = mnist_train[new_idx]
mnist_train_single_labels=np.asarray(mnist_train_single_labels)[new_idx]
# partition testing set into +/- groups: ratio=(7:3)
mnist_test_single_labels=[]
for i in range(np.shape(mnist.test.labels)[0]):
    if mnist.test.labels[i]>4:
        mnist_test_single_labels.append(1)
    else:
        mnist_test_single_labels.append(0)
# as np array
mnist_test_single_labels=np.asarray(mnist_test_single_labels)

#W_range=50.0
batch_size = 32
# AUC neural net model
class AUCModel(object):
    global p    
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        
        self.X = tf.placeholder(tf.float32, [None,28,28,1])
        self.y_sing = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('feature_extraction'):
            #CNN layers:
            self.W_conv0 = tf.get_variable("W_conv0",[5,5,1,32],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))
            self.b_conv0 = tf.get_variable("b_conv0",[32],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))
            self.h_conv0 = conv2d_stride(self.X, self.W_conv0, 1) + self.b_conv0
            self.bn_conv0 = tf.contrib.layers.batch_norm(self.h_conv0, 
                                                         center=True, scale=True, 
                                                         scope='bn0')

            self.W_conv1 = tf.get_variable("W_conv1",[5,5,32,256],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))
            self.b_conv1 = tf.get_variable("b_conv1",[256],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))
            self.h_conv1 = conv2d_stride(self.bn_conv0, self.W_conv1, 1) + self.b_conv1
            self.bn_conv1 = tf.contrib.layers.batch_norm(self.h_conv1, 
                                                         center=True, scale=True,
                                                         scope='bn1')
             
            self.W_conv2 = tf.get_variable("W_conv2",[3,3,256,1024],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))
            self.b_conv2 = tf.get_variable("b_conv2",[1024],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,1.0))
            self.h_conv2 = conv2d_stride(self.bn_conv1, self.W_conv2, 2) + self.b_conv2
            print('h_conv2.shape',self.h_conv2.shape)
            self.bn_conv2 = tf.contrib.layers.batch_norm(self.h_conv2, 
                                                         center=True, scale=True, 
                                                         scope='bn2')
            # The feature vector
            self.feature = tf.reshape(self.bn_conv1, [-1, 14*14*1024])
            self.w = tf.Variable(1e-5*tf.ones([14*14*1024, 1],dtype=tf.float32),name='w',trainable=False)
            self.inner_prod=tf.matmul(self.feature,self.w)
            self.pred = 0.5*tf.sign(self.inner_prod)+0.5
        
        with tf.variable_scope('network'):
            self.a = tf.Variable(tf.zeros([1],dtype=tf.float32),name='a')
            self.b = tf.Variable(tf.zeros([1],dtype=tf.float32),name='b')
            self.alpha = tf.Variable(tf.zeros([1],dtype=tf.float32),name='alpha')
            
            self.loss1 = ( 1 - p ) * tf.multiply( tf.square( self.inner_prod - tf.tile( self.a, [batch_size] ) ), self.y_sing )
            self.loss2 = p * tf.multiply( tf.square( self.inner_prod - tf.tile( self.b, [batch_size] ) ), 1 - self.y_sing )
            self.loss3 = 2 * ( 1 + self.alpha ) * ( p * tf.multiply( self.inner_prod, ( 1 - self.y_sing ) ) \
                                                    - ( 1 - p ) * tf.multiply( self.inner_prod, self.y_sing ) ) \
                         - p * ( 1 - p ) * tf.square( self.alpha )
            self.loss = tf.reduce_mean( self.loss1 + self.loss2 + self.loss3 )

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = AUCModel()
    
    learning_rate = tf.placeholder(tf.float32, [])
    
    assign_a_op = tf.assign(model.a, tf.reshape( tf.reduce_mean( tf.multiply( model.inner_prod,  model.y_sing ) ), [1] ) )
    assign_b_op = tf.assign(model.b, tf.reshape( tf.reduce_mean( tf.multiply( model.inner_prod, 1 - model.y_sing ) ), [1] ) )
    assign_alpha_op = tf.assign(model.alpha, tf.reshape( tf.reduce_mean( tf.multiply(model.inner_prod, 1 - 2 * model.y_sing ) ), [1] ) )

    t_vars = tf.trainable_variables()
    feat_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # feature layer optimization:
    feat_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_extraction')
    grads_and_vars_feat = feat_train_op.compute_gradients( model.loss, feat_vars)
    feat_op = feat_train_op.apply_gradients( grads_and_vars_feat)
    critic_op = tf.group(assign_a_op, assign_b_op, assign_alpha_op)
    actor_op = tf.group(feat_op)
    train_op = tf.group(critic_op, actor_op)
    # Evaluation
    correct_label_pred = tf.equal(tf.cast(model.y_sing,tf.int32), tf.cast(tf.transpose(model.pred), tf.int32))
    label_accuracy = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))

# Params
num_steps = 50000#FLAGS.num_time_steps

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        
        gen_train_batch = batch_generator(
            [mnist_train, mnist_train_single_labels], batch_size)
        gen_test_batch = batch_generator(
            [mnist_test, mnist_test_single_labels], batch_size)
        # Training loop
        wc=0.0
        for i in range(num_steps):
            # Learning rate as in Stochastic Online AUC optimization
            X, y_sing = gen_train_batch.next()
            X_t, y_sing_t = gen_test_batch.next()
            
            lr = 2e-5/np.sqrt(i+1)
            wc = wc + lr
            
            #Training step
            accuracy, batch_total_loss, correct_label_prediction, prediction, A, B, Alpha, inner_product, feature, __ = \
            sess.run([label_accuracy, model.loss, correct_label_pred, model.pred, model.a, model.b, model.alpha, model.inner_prod, model.feature,  train_op],
                       feed_dict = {model.X: X, model.y_sing: y_sing, learning_rate: lr})
                    
            acc_test = sess.run(label_accuracy,
                            feed_dict = {model.X: X_t, model.y_sing: y_sing_t})
            if verbose and i % 200 == 199:
                print '\n\nAUC optimization training, (+/-) ratio = 5:5'
                print 'epoch',i,'/',num_steps 
                AUC=tf.contrib.metrics.streaming_auc(prediction, y_sing)
                sess.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
                train_AUC=sess.run(AUC)
                print 'batch_total_loss',batch_total_loss
                print 'A',A
                print 'B',B
                print 'Alpha',Alpha
                print 'train_auc',train_AUC
                print 'train_acc',accuracy
                print 'test_acc',acc_test
                print 'inner_product',inner_product.T
                print 'prediction ', prediction.reshape([batch_size]).astype(int)
                print 'groundtruth', y_sing.astype(int)
                print 'correct    ', correct_label_prediction.reshape([batch_size]).astype(int)
            
        # Compute final evaluation on test data
        #acc = sess.run(label_accuracy,
        #                    feed_dict={model.X: mnist_test, model.y_sing: mnist_test_single_labels})
    #return acc #, train_auc, train_pre, train_rec


print '\nauc optimization training'
train_and_evaluate('auc', graph, model)
#print 'auc training accuracy:', acc
