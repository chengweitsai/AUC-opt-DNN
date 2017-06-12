from __future__ import division

import tensorflow as tf
import numpy as np
from utils import *
from sklearn.externals.joblib import Memory
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
flags.DEFINE_integer("num_epochs", 25, "number of times to repeat the same experiment")
FLAGS = flags.FLAGS


###################################################

#print 'process dataset covtype.libsvm.binary.scale/HIGGS/skin_nonskin/SUSY/real-sim /ijcnn1/phishing'


#mem = Memory("./mycache")
#@mem.cache
def get_data():
        data = load_svmlight_file('./data/'+FLAGS.dataset)
        return data[0], data[1]


print 'load data'
X,y = get_data()
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
            y_new.append(1)
            X_new.append(X[i])
        elif neg_count % np.ceil(C)==0:
            neg_count += 1
            y_new.append(0)
            X_new.append(X[i])
        else:
            neg_count +=1
else:
    for i in range(data_num):
        if y[i]!=1:
            neg_count += 1
            y_new.append(0)
            X_new.append(X[i])
        elif pos_count % np.ceil(1/C)==0:
            pos_count += 1
            y_new.append(1)
            X_new.append(X[i])
        else:
            pos_count +=1

X_new=np.squeeze(np.array(X_new))
print 'X_new.shape', X_new.shape
y_new=np.array(y_new)
print 'y_new.shape', y_new.shape
feat_num=np.shape(X_new)[1]
X_new_mean=np.mean(X_new,axis=0)
new_data_num=np.shape(X_new)[0]
"""
X_mean=np.mean(X,axis=0)
# mean 0
X = X - np.stack([X_mean for _ in range(data_num)])
# norm 1
for i in range(data_num):
    X[i,:]=X[i,:]/np.linalg.norm(X[i,:])
p=np.mean(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
X_new_mean=np.mean(X_new,axis=0)
# mean 0
X_new = X_new - np.stack([X_new_mean for _ in range(new_data_num)])
# norm 1
for i in range(new_data_num):
    X_new[i,:]=X_new[i,:]/np.linalg.norm(X_new[i,:])
p=np.mean(y_new)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
print 'X_train shape',X_train.shape
print 'X_test shape',X_test.shape
new_idx = np.random.permutation(np.shape(y_train)[0])
# shuffle training set:
X_train = X_train[new_idx]
y_train=np.asarray(y_train)[new_idx]
train_num = X_train.shape[0]
test_num = X_test.shape[0]
#---------------------------------------------------------------------------
W_range=1.0
batch_size = FLAGS.batch_size
# AUC neural net model
class AUCModel(object):
    global p    
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        
        self.X = tf.placeholder(tf.float64, [None,feat_num])
        self.y_sing = tf.placeholder(tf.float64, [None])
        
        with tf.variable_scope('weight'):
            # current copy of w 
            self.w = tf.Variable(tf.zeros([feat_num,1],dtype=tf.float64), name="w")
            # average version of w
            self.w_ave = tf.Variable(tf.zeros([feat_num,1],dtype=tf.float64), name="w_ave")
            self.inner_prod=tf.matmul(self.X,self.w)
            self.inner_prod_ave=tf.matmul(self.X,self.w_ave)
            self.pred = 0.5*tf.sign(self.inner_prod)+0.5
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
    # grads_and_vars is a list of tuples (grad,var)
    #grads_min, vars_min = zip(*grads_and_vars_min)
    #clipped_grads_and_vars_min = [(tf.clip_by_norm(grad_min,clip_norm=500),var_min) for (grad_min,var_min) in grads_and_vars_min]
    #min_op = min_train_op.apply_gradients(clipped_grads_and_vars_min)
    min_op = min_train_op.apply_gradients(grads_and_vars_min)
    
    clip_a_op=tf.assign(model.a,tf.clip_by_value(model.a, clip_value_min=-W_range, clip_value_max=W_range))
    clip_b_op=tf.assign(model.b,tf.clip_by_value(model.b, clip_value_min=-W_range, clip_value_max=W_range))
    clip_w_op=tf.assign(model.w,tf.clip_by_norm(model.w, clip_norm = W_range,axes=[0]))
    # stochastic ascent
    # compute the gradients of a list of vars: alpha
    grads_and_vars_max=max_train_op.compute_gradients(tf.negative(model.loss),[v for v in t_vars if v.name=='network/alpha:0'])
    # grads_and_vars is a list of tuples (grad,var)
    #grads_max, vars_max = zip(*grads_and_vars_max)
    #clipped_grads_and_vars_max = [(tf.clip_by_norm(grad_max,clip_norm=500),var_max) for (grad_max,var_max) in grads_and_vars_max]
    #max_op = min_train_op.apply_gradients(clipped_grads_and_vars_max)
    max_op = min_train_op.apply_gradients(grads_and_vars_max)
                
    clip_alpha_op=tf.assign(model.alpha,tf.clip_by_value(model.alpha, clip_value_min=-2*W_range, clip_value_max=2*W_range))
    

    train_op = tf.group(max_op, clip_alpha_op, min_op, clip_a_op, clip_b_op, clip_w_op,
                        save_w_op, save_a_op, save_b_op, save_alpha_op)
    # Evaluation
    correct_label_pred = tf.equal(tf.cast(model.y_sing,tf.int64), tf.cast(tf.transpose(model.pred),tf.int64))
    label_accuracy = tf.reduce_mean(tf.cast(correct_label_pred, tf.float64))

# Params
num_steps = 50000

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""
    wc=0
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        #global X_train, X_test, y_train, y_test       
        gen_train_batch = batch_generator(
            [X_train, y_train], batch_size)
        gen_test_batch = batch_generator(
            [X_test, y_test], batch_size)
        # Training loop
        prediction_list = []
        label_list = []
        for i in range(num_steps):
            # Learning rate as in Stochastic Online AUC optimization
            
            # Training step
            if training_mode == 'auc':
                lr = 1e1/np.sqrt(i+1)
                wc = wc + lr
                #lr = 0
                X, y_sing = gen_train_batch.next()
                # fetch loss
                accuracy, batch_total_loss, prediction, frac, W, A, B, Alpha, inner_product, gvmin, gvmax, correct,  _ = \
                               sess.run([label_accuracy, model.loss, model.pred, fraction, model.w, model.a, model.b, model.alpha, \
                                         model.inner_prod_ave, grads_and_vars_min, grads_and_vars_max, correct_label_pred, train_op], \
                               feed_dict={model.X: X, model.y_sing: y_sing, learning_rate: lr,weighted_coeff: wc})
                prediction_list.extend(prediction.reshape([batch_size]))
                label_list.extend(y_sing)
                #print(np.array(label_list).shape)
                #print(np.array(prediction_list).shape)
                if verbose and i % 2000 == 1999:
                    print '\n\nAUC optimization training, (+/-) ratio %f', p,1-p
                    print 'epoch',i,'/',num_steps
                    AUC=tf.contrib.metrics.streaming_auc(prediction, y_sing)
                    sess.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
                    train_AUC=sess.run(AUC)
                    gmin,vmin=zip(*gvmin)
                    gmax,vmax=zip(*gvmax)
                    print 'batch_total_loss',batch_total_loss
                    print 'learning_rate',lr
                    print 'fraction',frac
                    #print 'W',W.T
                    #print 'gmin length',len(gmin)
                    #print 'gmin[0].shape',gmin[0].shape
                    #print 'gradient_w',gmin[0]
                    #print 'gmin[1].shape',gmin[1].shape
                    print 'gradient_a',gmin[1]
                    #print 'gmin[2].shape',gmin[2].shape
                    print 'gradient_b',gmin[2]
                    #print 'gvmax length',len(gvmax)
                    #print 'gmax[0].shape',gmax[0].shape
                    print 'gradient_alpha',gmax[0]
                    print 'A',A
                    print 'B',B
                    print 'Alpha',Alpha
                    #print('weighted_coeff',weight)
                    print 'train_auc',train_AUC
                    print 'train_acc',accuracy
                    #cumulative_AUC = metrics.roc_auc_score(np.array(label_list),np.array(prediction_list))
                    #print 'cumulative AUC', cumulative_AUC
                    print 'inner_product',inner_product.T
                    print 'prediction ', prediction.reshape([batch_size]).astype(int)
                    print 'groundtruth', y_sing.astype(int)
                    print 'correct    ', correct.reshape([batch_size]).astype(int)
            
        # Compute final evaluation on test data

        acc ,prediction = sess.run([label_accuracy, model.pred],
                            feed_dict={model.X: X_test, model.y_sing: y_test})
        test_prediction = prediction.reshape([test_num])
        cumulative_auc = metrics.roc_auc_score(y_test , test_prediction)
    return acc ,cumulative_auc#, train_auc, train_pre, train_rec
if not FLAGS.dataset:
    raise ValueError("Must set --dataset for experiments")
if not FLAGS.output_file:
    raise ValueError("Must set --output_file for experiments")
fout =open('./output/'+FLAGS.output_file,'a')
fout.write('dataset: '+FLAGS.dataset)
fout.write('\noutput_file: '+FLAGS.output_file)
fout.write('\n(+/-) ratio: '+str(FLAGS.request_ratio)+':'+str(1-FLAGS.request_ratio))
fout.write('\nAUC optimization with '+str(FLAGS.num_time_steps)+ ' training steps')
fout.close()
print 'dataset:', FLAGS.dataset
print 'output_file:', FLAGS.output_file
print '(+/-) ratio:', FLAGS.request_ratio,' : ',1-FLAGS.request_ratio
print '\nauc optimization training'
auc_ave=0.0
for i in range(FLAGS.num_epochs):
    print 'epoch', i,'/',str(FLAGS.num_epochs)
    fopen =open('./output/'+FLAGS.output_file,'a')
    fopen.write('\nNumber of experiment '+str(i)+' / '+str(FLAGS.num_epochs)+'\n')
    acc,auc = train_and_evaluate('auc', graph, model)
    print 'testing data accuracy:', acc
    fopen.write('testing data accuracy: '+str(acc)+'\n')
    print 'testing data cumulative AUC', auc
    fopen.write('testing data culumative AUC: '+str(auc)+'\n')
    auc_ave= (i*auc_ave+auc)/((i+1)*1.0)
    fopen.close()
fopen =open('./output/'+FLAGS.output_file,'a')
fopen.write('testing data average_culumative AUC over '+str(FLAGS.num_epochs)+' epochs: '+str(auc_ave)+'\n')
fopen.close()
