import numpy as np
import random
import os
import sys
sys.path.append(os.pardir)
sys.path.append('c:/Users/bueno/Documents/Deeplearning/HomeWork/YOURSTUDENTNUMBER')

import matplotlib.pyplot as plt
import _pickle as pickle
import time
from YourAnswer import naive_softmax_loss, vectorized_softmax_loss
from YourAnswer import Softmax

# set default plot options
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load all of CIFAR10 dataset.
def load_CIFAR10(root):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(root, 'data_batch_%d' % (b, ))
        with open(f, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
        #X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X,Y
    
    f=os.path.join(root, 'test_batch')
    with open(f, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        Xte = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        Yte = np.array(Y)
        
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data():
    # 1. Load the raw data
    X_tr, Y_tr, X_te, Y_te = load_CIFAR10('c:/Users/bueno/Documents/Deeplearning/HomeWork/cifar-10-batches-py')
    
    # 2. Divide the data
    X_val, Y_val = X_tr[49000:], Y_tr[49000:]
    X_tr, Y_tr = X_tr[:49000], Y_tr[:49000]
    X_te, Y_te = X_te[:1000], Y_te[:1000]

    # 3. Preprocess the input image
    X_tr = np.reshape(X_tr, (X_tr.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0],-1))
    X_te = np.reshape(X_te, (X_te.shape[0],-1))
    
    # 4. Normalize the data (subtract the mean image)
    mean_img = np.mean(X_tr, axis = 0)
    X_tr -= mean_img
    X_val -= mean_img
    X_te -= mean_img

    # 5. Add bias and Transform into columns
    X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0],1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0],1))])
    X_te = np.hstack([X_te, np.ones((X_te.shape[0],1))])
    
    return X_tr, Y_tr, X_val, Y_val, X_te, Y_te, mean_img



X_tr, Y_tr, X_val, Y_val, X_te, Y_te, mean_img = get_CIFAR10_data()
print ('Train data shape : %s,  Train labels shape : %s' % (X_tr.shape, Y_tr.shape))
print ('Validatoin data shape : %s,  Validation labels shape : %s' % (X_val.shape, Y_val.shape))
print ('Test data shape : %s,  Test labels shape : %s' % (X_te.shape, Y_te.shape))


class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

images_index = np.int32(np.round(np.random.rand(18,)*49000,0))

fig, axes = plt.subplots(3, 6, figsize=(18, 6),
                         subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(hspace=0.3, wspace=0.05)

# for ax, idx in zip(axes.flat, images_index):
#     img = (X_tr[idx,:3072].reshape(32, 32, 3) + mean_img.reshape(32, 32, 3))/255.
#     ax.imshow(img)
#     ax.set_title(class_names[Y_tr[idx]])

# 1. Softmax Classifier

# W = np.random.randn(3073, 10) * 0.0001
# loss, grad = naive_softmax_loss(W, X_tr, Y_tr, 0.0)

# print ('loss :', loss)
# print ('sanity check : ', -np.log(0.1))

# s_time = time.time()
# loss_naive, grad_naive = naive_softmax_loss(W, X_tr, Y_tr, 0.00001)
# print ('naive loss : %e with %fs' % (loss_naive, time.time()-s_time))

# s_time = time.time()
# loss_vectorized, grad_vectorized = vectorized_softmax_loss(W, X_tr, Y_tr, 0.00001)
# print ('vectorized loss : %e with %fs' % (loss_vectorized, time.time()-s_time))

# print ('loss difference : %f' % np.abs(loss_naive - loss_vectorized))
# print ('gradient difference : %f' % np.linalg.norm(grad_naive-grad_vectorized, ord='fro'))


# results is dictionary mapping tuples of the form.
# (learning_rate, regularization_strength) to tuple of the form (training_accuracy, validation_accuracy).
# The accuracy is simply the fraction of data points that are correctly classified.
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-8, 1e-7, 5e-7, 1e-6]
regularization_strengths = [5e2, 1e3, 1e4, 5e4]
train_acc = 0
val_acc = 0
#########################################################################################################
# TODO : Write code that chooses the best hyperparameters by tuning on the validation set.              # 
#        For each combination of hyperparemeters, train a Softmax on the training set,                  #
#        compute its accuracy on the training and validatoin sets, and store these numbers in the       #
#        results dictionary. In addition, store the best validation accuracy in best_val                #
#        and the Softmax object that achieves this accuracy in best_softmax.                            #
#                                                                                                       #
# Hint : You should use a small value for num_iters as you develop your validation code so that the     #
#        Softmax don't take much time to train; once you are confident that your validation code works, #
#        you should rerun the validation code with a larger value for num_iter.                         #

#softmax = Softmax()
        
for l_rate in learning_rates:
    for reg in regularization_strengths:
#------------------------------------------WRITE YOUR CODE----------------------------------------------#
        softmax = Softmax()
        iterations=100 #default 100
        bs=128 #default 128
        softmax.train(X_tr, Y_tr, X_val, Y_val,l_rate,reg,iterations,bs)
        
        y_train_pred = softmax.predict(X_tr)
        train_acc = softmax.get_accuracy(X_tr,Y_tr)
        y_val_pred = softmax.predict(X_val)
        val_acc = softmax.get_accuracy(X_val,Y_val) 
 
        if best_val < val_acc:
            best_val = val_acc
            best_softmax = softmax

#-----------------------------------------END OF YOUR CODE----------------------------------------------#
#########################################################################################################
        results[(l_rate,reg)] = (train_acc, val_acc)
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy : %f, val accuracy : %f ' % (lr, reg, train_accuracy, val_accuracy))
    
print ('best validatoin accuracy achieved during cross-validation :', best_val)


Y_te_pred = best_softmax.predict(X_te)
# test_accuracy = softmax.get_accuracy(X_te,Y_te)
test_accuracy = np.mean(Y_te == Y_te_pred)
print ('softmax on raw pixels final test set accuracy : ', test_accuracy)