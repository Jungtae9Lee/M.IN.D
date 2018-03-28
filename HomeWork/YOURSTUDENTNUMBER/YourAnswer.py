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

            #--------------------------------------END OF YOUR CODE--------------------------------------------#
            ####################################################################################################

            # Evaluate loss and gradient
            tr_loss, tr_grad = self.loss(X_batch, Y_batch, reg)

            # Perform parameter update
            ####################################################################################################
            # TODO : Update the weights using the gradient and the learning rate                               #
            #---------------------------------------WRITE YOUR CODE--------------------------------------------#

            #--------------------------------------END OF YOUR CODE--------------------------------------------#
            ####################################################################################################

            if verbose and it % num_iters == 0:
                print ('Ieration %d / %d : loss %f ' % (it, num_iters, loss))
            
        
    
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
        accuracy = 0  ####################################################################################################
        # TODO : Implement this method. Calculate an accuracy of X_data using Y_data and predict Func                               #
        #---------------------------------------WRITE YOUR CODE--------------------------------------------#

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
    #        Don't forget the regularization.                                                          #
    #---------------------------------------WRITE YOUR CODE--------------------------------------------#

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

    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    ####################################################################################################
    
    return softmax_loss, dWeights









