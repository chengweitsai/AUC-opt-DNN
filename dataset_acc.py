from __future__ import division

import tensorflow as tf
import numpy as np
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import metrics

############# define input options ################
flags = tf.flags

flags.DEFINE_string(
    "dataset", None,
    "Choose one of the datasets: (1) phishing, (2) real-sim, (3) skin_nonskin, (4) SUSY, (5) epsilon_normalized, (6) covtype.libsvm.binary.scale, (7) ijcnn1, (8) HIGGS, (9) diabetes_scale, (10) heart_scale, (11) rcv1_train.binary")
flags.DEFINE_string("output_file", None,
                    "Where the training/test experiment data is stored.")
flags.DEFINE_float("request_ratio", 0.7, "Positive / Negative data ratio. Default value 0.7")
flags.DEFINE_integer("batch_size", 32, "batch_size (default = 32).")
flags.DEFINE_integer("num_time_steps", 50000, "number of time steps for the AUC optimization")
flags.DEFINE_integer("num_epochs", 10, "number of times to repeat the same experiment")
FLAGS = flags.FLAGS


###################################################

def get_data():
    data = load_svmlight_file('./data/'+FLAGS.dataset)
    return data[0], data[1]


print 'load data'
X,y = get_data()
print 'todense'
X = X.todense()
data_num, feat_num = np.shape(X)
# compute (+/-) ratio of dataset:
data_ratio=0
for i in range(data_num):
    if y[i]==1:
       data_ratio=(i*data_ratio+y[i])/((i+1)*1.0)
    else:
       y[i]=0
       data_ratio=(i*data_ratio+y[i])/((i+1)*1.0)
print 'data_ratio=',data_ratio

print 'relabel y=1/0 & reset (+/-) ratio:'
X_new=[]
y_new=[]
pos_count=0
neg_count=0
C=FLAGS.request_ratio*(1-data_ratio)/(data_ratio*(1-FLAGS.request_ratio))
if FLAGS.request_ratio > data_ratio:
    for i in range(data_num):
        if y[i]==1:
            pos_count += 1
            y_new.append([1,0])
            X_new.append(X[i,:])
        elif neg_count % np.ceil(C)==0:
            neg_count += 1
            y_new.append([0,1])
            X_new.append(X[i,:])
        else:
            neg_count +=1
else:
    for i in range(data_num):
        if y[i]!=1:
            neg_count += 1
            y_new.append([0,1])
            X_new.append(X[i,:])
        elif pos_count % np.ceil(1/C)==0:
            pos_count += 1
            y_new.append([1,0])
            X_new.append(X[i,:])
        else:
            pos_count +=1

X_new=np.squeeze(np.array(X_new))
print 'X_new.shape', X_new.shape
y_new=np.array(y_new)
print 'y_new.shape', y_new.shape
new_data_num,feat_num = np.shape(X_new)
# mean 0
X_new_mean = np.mean(X_new,axis=0)
X_new = X_new - np.stack([X_new_mean for _ in range(new_data_num)])
# norm 1
for i in range(new_data_num):
    X_new[i,:]=X_new[i,:]/np.linalg.norm(X_new[i,:])
p=np.mean(y_new)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
print 'X_train shape',X_train.shape
print 'X_test shape',X_test.shape
# shuffle training set:
new_idx = np.random.permutation(np.shape(y_train)[0])
X_train = X_train[new_idx]
y_train=np.asarray(y_train)[new_idx]
train_num = X_train.shape[0]
test_num = X_test.shape[0]

#---------------------------------------------------------------------------
batch_size = FLAGS.batch_size
# ACC shallow neural net model
class ACCModel(object):
    
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        
        self.X = tf.placeholder(tf.float32, [None, feat_num])
        self.y_bin = tf.placeholder(tf.float32, [None, 2])
        
        with tf.variable_scope('weight'):
            #self.w 
            self.w = tf.get_variable("w",[feat_num, 2],dtype=tf.float32,initializer=tf.random_normal_initializer(0.0,0.01))
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
num_steps = FLAGS.num_time_steps

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        
        gen_train_batch = batch_generator(
            [X_train, y_train], batch_size)
        gen_test_batch = batch_generator(
            [X_test, y_test], batch_size)
        # Training loop
        for i in range(num_steps):
            
            if training_mode == 'acc':
                lr = 1e-2/np.sqrt(i+1)
                X, y_bin = gen_train_batch.next()
                y_single=y_bin[:,1]
                _, W, accuracy, batch_pred_loss, pred, prob = sess.run([regular_train_op, model.w, label_accuracy, model.pred_loss, max_pred, model.pred],
                                     feed_dict={model.X: X, model.y_bin: y_bin,learning_rate: lr})
                if verbose and i % 1000 == 0:
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
        acc, test_prediction = sess.run([label_accuracy,max_pred],
                            feed_dict={model.X: X_test, model.y_bin: y_test})
        print "y_test",y_test.shape
        print "test_prediction",test_prediction.shape
        cumulative_auc = metrics.roc_auc_score(y_test[:,1] , test_prediction)
    return acc, cumulative_auc


print '\nacc only training'
acc = train_and_evaluate('acc', graph, model)
print 'acc training accuracy:', acc


if not FLAGS.dataset:
    raise ValueError("Must set --dataset for experiments")
if not FLAGS.output_file:
    raise ValueError("Must set --output_file for experiments")
fout =open('./output/'+FLAGS.output_file,'a')
fout.write('dataset: '+FLAGS.dataset)
fout.write('\noutput_file: '+FLAGS.output_file)
fout.write('\n(+/-) ratio: '+str(FLAGS.request_ratio)+':'+str(1-FLAGS.request_ratio))
fout.write('\nACC optimization with '+str(FLAGS.num_time_steps)+ ' training steps')
fout.close()
print 'dataset:', FLAGS.dataset
print 'output_file:', FLAGS.output_file
print '(+/-) ratio:', FLAGS.request_ratio,' : ',1-FLAGS.request_ratio
print '\nauc optimization training'
acc_ave=0.0
auc_ave=0.0
for i in range(FLAGS.num_epochs):
    print 'epoch', i,'/',str(FLAGS.num_epochs)
    fopen =open('./output/'+FLAGS.output_file,'a')
    fopen.write('\nNumber of experiment '+str(i)+' / '+str(FLAGS.num_epochs)+'\n')
    acc,auc = train_and_evaluate('acc', graph, model)
    print 'testing data accuracy:', acc
    fopen.write('testing data accuracy: '+str(acc)+'\n')
    print 'testing data cumulative AUC', auc
    fopen.write('testing data culumative AUC: '+str(auc)+'\n')
    acc_ave= (i*acc_ave+acc)/((i+1)*1.0)
    auc_ave= (i*auc_ave+auc)/((i+1)*1.0)
    fopen.close()
fopen =open('./output/'+FLAGS.output_file,'a')
fopen.write('testing data average ACC over '+str(FLAGS.num_epochs)+' epochs: '+str(acc_ave)+'\n')
fopen.write('testing data average AUC over '+str(FLAGS.num_epochs)+' epochs: '+str(auc_ave)+'\n')
fopen.close()
