from __future__ import division

import tensorflow as tf
import numpy as np
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

############# define input options ################
flags = tf.flags

flags.DEFINE_string(
    "dataset", None,
    "Choose one of the datasets: (1) phishing, (2) real-sim, (3) skin_nonskin, (4) SUSY, (5) epsilon_normalized, (6) covtype.libsvm.binary.scale, (7) ijcnn1, (8) HIGGS, (9) diabetes_scale, (10) heart_scale, (11) rcv1_train.binary")
flags.DEFINE_string("output_file", None,
                    "Where the training/test experiment data is stored.")
flags.DEFINE_float("request_ratio", 0.7, "Positive / Negative data ratio. Default value 0.7")
flags.DEFINE_integer("batch_size", 32, "batch_size (default = 32).")
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
       y[i]=0 #y=-1
       data_ratio=(i*data_ratio+y[i])/((i+1)*1.0)
print 'data_ratio=',data_ratio

print 'relabel y=1/0 & reset (+/-) ratio:'
X_new=np.array([]).reshape(-1, X.shape[1])
y_new=np.array([])
X_pos = X[y==1].reshape(-1, X.shape[1])
X_neg = X[y==0].reshape(-1, X.shape[1])

C=FLAGS.request_ratio*(1-data_ratio)/(data_ratio*(1-FLAGS.request_ratio))
if FLAGS.request_ratio > data_ratio:
    X_new = np.r_[X_new, X_pos]
    y_new = np.r_[y_new, np.ones(X_pos.shape[0])]
    neg_idx = np.arange(0, X_neg.shape[0], np.ceil(C)).astype(np.int32)
    X_new = np.r_[X_new, X_neg[neg_idx].reshape(-1, X.shape[1])]
    y_new = np.r_[y_new, np.zeros(neg_idx.size)]
else:
    X_new = np.r_[X_new, X_neg]
    y_new = np.r_[y_new, np.zeros(X_neg.shape[0])]
    pos_idx = np.arange(0, X_pos.shape[0], np.ceil(C)).astype(np.int32)
    X_new = np.r_[X_new, X_pos[pos_idx].reshape(-1, X.shape[1])]
    y_new = np.r_[y_new, np.ones(pos_idx.size)]

print 'X_new.shape', X_new.shape
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
y_train =  y_train[new_idx]
train_num = X_train.shape[0]
test_num = X_test.shape[0]

def train_and_evaluate():
    """Helper to run the model with different training modes."""

    # solving logistic regression 
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # make predictions
    test_prediction = model.predict(X_test)
    acc = 1.0 * np.sum(y_test==test_prediction) / y_test.size

    # Compute final evaluation on test data
    print "y_test",y_test.shape
    print "test_prediction",test_prediction.shape
    cumulative_auc = metrics.roc_auc_score(y_test , test_prediction)
    return acc, cumulative_auc

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
print '\nacc optimization training'
acc_ave,auc_ave = train_and_evaluate()
fopen =open('./logistic/'+FLAGS.output_file,'a')
fopen.write('testing data ACC: '+str(acc_ave)+'\n')
fopen.write('testing data AUC: '+str(auc_ave)+'\n')
fopen.close()
print 'testing data ACC: '+str(acc_ave)
print 'testing data AUC: '+str(auc_ave)
