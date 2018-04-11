import numpy as np
import random

class Softmax(object):
    def __init__(self):
        #self.Weights = None
        return
        
    def train(self, X_tr_data, Y_tr_data, X_val_data, Y_val_data, lr=1e-3, reg=1e-5, iterations=100, bs=128, verbose=False, weight=0):
        """
        Train this Softmax classifier using stochastic gradient descent.
        
        Inputs have D dimensions, and we operate on N examples.
        
        Inputs :
            - X_data : A numpy array of shape (N,D) containing training data.
            - Y_data : A numpy array of shape (N,) containing training labels;
                  Y[i]=c means that X[i] has label 0<=c<C for C classes.
            - lr : (float) Learning rate for optimization.
            - reg : (float) Regularization strength. 
            - iterations : (integer) Number of steps to take when optimizing. 
            - bs : (integer) Number of training examples to use at each step.
            - verbose : (boolean) If true, print progress during optimization.
        
        Regurns :
            - A list containing the value of the loss function at each training iteration.
        """
        
        num_train, dim = X_tr_data.shape
        num_classes = np.max(Y_tr_data)+1
        self.Weights = 0.001*np.random.randn(dim, num_classes)
        
        if np.shape(weight)!=np.shape(0):
            self.Weights = weight
            
        for it in range(iterations):
            #X_batch = None
            #Y_batch = None
            
            ####################################################################################################
            # TODO : Sample batch_size elements from the training data and their corresponding labels          #
            #        to use in this round of gradient descent.                                                 #
            #        Store the data in X_batch and their corresponding labels in Y_batch; After sampling       #
            #        X_batch should have shape (dim, batch_size) and Y_batch should have shape (batch_siae,)   #
            #                                                                                                  #
            #        Hint : Use np.random.choice to generate indicies.                                         #
            #               Sampling with replacement is faster than sampling without replacement.             #
            #---------------------------------------WRITE YOUR CODE--------------------------------------------#
            batch_idx = np.random.choice(num_train, bs, replace = True)
            X_batch =  X_tr_data[batch_idx]
            Y_batch = Y_tr_data[batch_idx]
            #--------------------------------------END OF YOUR CODE--------------------------------------------#
            ####################################################################################################

            # Evaluate loss and gradient
            loss_history = []
            tr_loss, tr_grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(tr_loss)

            # Perform parameter update
            ####################################################################################################
            # TODO : Update the weights using the gradient and the learning rate                               #
            #---------------------------------------WRITE YOUR CODE--------------------------------------------#
            self.Weights += - lr * tr_grad
            #--------------------------------------END OF YOUR CODE--------------------------------------------#
            ####################################################################################################

            # if verbose and it % num_iters == 0:
            #     print ('Ieration %d / %d : loss %f ' % (it, num_iters, loss))
                    
    
    def predict(self, X_data):
        """
        Use the trained weights of this softmax classifier to predict labels for data points.
        
        Inputs :
            - X : A numpy array of shape (N,D) containing training data.
            
        Returns :
             - Y_pred : Predicted labels for the data in X. Y_pred is a 1-dimensional array of length N, 
                        and each element is an integer giving the predicted class.
        """
        Y_pred = np.zeros(X_data.shape[0])
        
        ####################################################################################################
        # TODO : Implement this method. Store the predicted labels in Y_pred                               #
        #---------------------------------------WRITE YOUR CODE--------------------------------------------#
        scores = X_data.dot(self.Weights)
        Y_pred = np.argmax(scores, axis = 1)
        #--------------------------------------END OF YOUR CODE--------------------------------------------#
        ####################################################################################################
        return Y_pred
    
    def get_accuracy(self, X_data, Y_data):
        """
        Use X_data and Y_data to get an accuracy of the model.
        
        Inputs :
            - X_data : A numpy array of shape (N,D) containing input data.
            - Y_data : A numpy array of shape (N,) containing a true label.
            
        Returns :
             - Accuracy : Accuracy of input data pair [X_data, Y_data].
        """
        accuracy = 0  
        ####################################################################################################
        # TODO : Implement this method. Calculate an accuracy of X_data using Y_data and predict Func                               #
        #---------------------------------------WRITE YOUR CODE--------------------------------------------#
        z = self.predict(X_data)
        # z_modify = np.argmax(z, axis=0)
       
        accuracy = np.sum(z == Y_data) / float(X_data.shape[0])
        #--------------------------------------END OF YOUR CODE--------------------------------------------#
        ####################################################################################################
        
        return accuracy
    
    def loss(self, X_batch, Y_batch, reg):
        return vectorized_softmax_loss(self.Weights, X_batch, Y_batch, reg)
    
    



def naive_softmax_loss(Weights,X_data,Y_data,reg):
    """
     Inputs have D dimension, there are C classes, and we operate on minibatches of N examples.
    
     Inputs :
         - Weights : A numpy array of shape (D,C) containing weights.
         - X_data : A numpy array of shape (N,D) contatining a minibatch of data.
         - Y_data : A numpy array of shape (N,) containing training labels; 
               Y[i]=c means that X[i] has label c, where 0<=c<C.
         - reg : Regularization strength. (float)
         
     Returns :
         - loss as single float
         - gradient with respect to Weights; an array of sample shape as Weights
     """
    
    # Initialize the loss and gradient to zero
    softmax_loss = 0.0
    dWeights = np.zeros_like(Weights)
    
    ####################################################################################################
    # TODO : Compute the softmax loss and its gradient using explicit loops.                           # 
    #        Store the loss in loss and the gradient in dW.                                            #
    #        If you are not careful here, it is easy to run into numeric instability.                  #
    #        Don't forget the regularization. ->                                                          #
    #---------------------------------------WRITE YOUR CODE--------------------------------------------#
    
    # 모든 원소가 0인 5x5리스트 생성
    # z_out = [[0]*np.size(X_data,0) for i in range(np.size(Weights,1))]
    # z_out = [[0]*np.size(Weights,1) for i in range(np.size(X_data,0))]
    # X[49000*3073] * W[3073*10]
    #z_out[49000*10]
    # too slow.. change loop only onetime
    # for i in range(np.size(X_data,0)): # 0~48999
    #     if(i % 100 == 0):
    #         print(i)
    #     for k in range(np.size(z_out,1)): #0~9
    #         for j in range(np.size(X_data,1)): #0~3072
    #             z_out[i][k] += X_data[i][j] * Weights[j][k]

       
    num_train = X_data.shape[0]
    num_classes = Weights.shape[1]

    for i in range(num_train):
        # Compute vector of scores
        f_i = X_data[i].dot(Weights)

        # Normalization trick to avoid numerical instability
        f_i -= np.max(f_i)

        # Compute softmax_loss (and add to it, divided later)
        sum_j = np.sum(np.exp(f_i))
        marco = lambda k: np.exp(f_i[k]) / sum_j
        softmax_loss += -np.log(marco(Y_data[i]))

        # Compute gradient
        # Here we are computing the contribution to the inner sum for a given i.
        for k in range(num_classes):
            p_k = marco(k)
            dWeights[:, k] += (p_k - (k == Y_data[i])) * X_data[i]

    softmax_loss /= num_train
    softmax_loss += 0.5 * reg * np.sum(Weights * Weights)
    dWeights /= num_train
    dWeights += reg*Weights
    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    ####################################################################################################
    return softmax_loss, dWeights



def vectorized_softmax_loss(Weights, X_data, Y_data, reg):
    softmax_loss = 0.0
    dWeights = np.zeros_like(Weights)

    ####################################################################################################
    # TODO : Compute the softmax loss and its gradient using no explicit loops.                        # 
    #        Store the loss in loss and the gradient in dW.                                            #
    #        If you are not careful here, it is easy to run into numeric instability.                  #
    #        Don't forget the regularization.                                                          #
    #---------------------------------------WRITE YOUR CODE--------------------------------------------#
    num_train = X_data.shape[0]
    f = X_data.dot(Weights)
    f -= np.max(f, axis=1, keepdims=True) # max of every sample
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f

    softmax_loss = np.sum(-np.log(p[np.arange(num_train), Y_data]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), Y_data] = 1
    dWeights = X_data.T.dot(p - ind)

    softmax_loss /= num_train
    softmax_loss += 0.5 * reg * np.sum(Weights * Weights)
    dWeights /= num_train
    dWeights += reg*Weights
    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    ####################################################################################################
    
    return softmax_loss, dWeights









