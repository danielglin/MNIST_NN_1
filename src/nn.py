# nn.py
# Daniel Lin, December 2018
# This script implements a neural network.

import numpy as np

class NNModel:
    def __init__(self, layers, activation_func, activation_func_grad,
                 lambd=0):
        """
        Builds a neural network
        
        Parameters:
            - layers: a list with the number of units in each layers
                The length determines the number of layers
                The first number in layers specifies the number of
                input units, which is also the number of features
            - activation_func: a callable object that is used as the 
                activation function for the hidden units
            - activation_func_grad: a callable object that is used as 
                the gradient of the activation function for the hidden units
            - lambd: the L2 regularization hyperparameter; defaults to 0
        """
        self.layers = layers
        self.activation_func = activation_func
        self.activation_grad = activation_func_grad
        self.weights = [0]
        self.biases = [0]
        self.cache = [0]
        self.Zs = [0]*len(layers)
        self.As = [0]*len(layers)
        self.Zs_grads = [0]*len(layers)
        self.weights_grads = [0]
        self.biases_grads = [0]
        self.lambd = lambd
        self.initialize_params()
    
    def initialize_params(self):
        """
        Implements He initialization for the weights
        """
        for i in range(1, len(self.layers)):
            self.weights.append(
                np.random.randn(self.layers[i], 
                                self.layers[i-1]) * np.sqrt(2/self.layers[i-1]))
            self.biases.append(np.zeros((self.layers[i], 1)))

            self.weights_grads.append(np.zeros(self.weights[i].shape))
            self.biases_grads.append(np.zeros(self.biases[i].shape))
        
        
    def forward_prop_single_layer(self, A_prev, num_layer):
        """
        Calculates the activation values for a hidden layer
        Assumes initialize_params has been called.
        
        Parameters:
            - A_prev: the activation values of the previous layer
            - num_layer: which layer to calculate the activation values for
            
        Returns:
            - A: the activation values for this layer
        """
        W = self.weights[num_layer]
        b = self.biases[num_layer]
        Z = W @ A_prev + b
        self.Zs[num_layer] = Z
        self.Zs_grads[num_layer] = np.zeros(Z.shape)
        A = self.activation_func(Z)
        self.As[num_layer] = A
        return A
        
    def forward_pass(self, X):
        """
        Performs a forward pass through the neural network
        Assumes initialize_params has been called.
        
        Parameters:
            - X: the input to the neural network
        Returns:
            - A_L: the activations of the last layer
        """
        self.As[0] = X
        
        # run softmax immediately if nn has 2 layers only
        if len(self.layers) - 1 == 1:
            A = X
        else:
            A = self.forward_prop_single_layer(X, 1)
        for i in range(2, len(self.layers)-1):
            A = self.forward_prop_single_layer(A, i)

        # use the output function for the output layer
        Z = self.weights[-1] @ A + self.biases[-1]
        self.Zs[-1] = Z
        A_L = self.softmax(Z)
        self.As[-1] = A_L
        return A_L
    
    def softmax(self, Z):
        """
        Implements softmax

        Parameters:
            - Z: a numpy array of shape (n, m)
            
        Returns:
            - a numpy array of shape (n, m) with softmax applied to each
                column of Z
        """
        CLIP_THRESHOLD = 700
        clipped_Z = np.clip(Z, a_min=None, a_max=CLIP_THRESHOLD)
        temp = np.exp(clipped_Z)
        A = temp / np.sum(temp, axis=0)
        return A
    
    def compute_cost(self, Y, A_L):
        """
        Computes the cross-entropy loss for softmax output layer
        
        Parameters:
            - Y: the true labels, has shape (C, m), where C is the 
                number of classes, and m is the number of examples
            - A_L: the activations of the output layer
            
        Returns:
            - cost: the cross-entropy loss
        """
        epsilon = 10e-8
        m = Y.shape[1]
        ln_A = np.log(A_L + epsilon)
        cost = -np.sum(np.diag(ln_A.T @ Y)) / m
        return cost
        
    def compute_cost_with_reglztn(self, Y, A_L):
        """
        Computes the cost including the cross-entropy loss and the 
        L2 regularization term
        
        Parameters:
            - Y: the true labels, has shape (C, m), where C is the 
                number of classes, and m is the number of examples
            - A_L: the activations of the output layer           
            
        Returns:
            - cost: the cross-entropy loss plus the
                L2 regularization term        
        """
        m = Y.shape[1]
        cross_entropy_cost = self.compute_cost(Y, A_L)
        regularized_cost = 0
        for i in range(1, len(self.layers)):
            regularized_cost += np.sum(np.square(self.weights[i]))          
        regularized_cost = 1./m * self.lambd/2. * regularized_cost
        cost = cross_entropy_cost + regularized_cost
        return cost
        
    def backward_prop(self, Y):
        """
        Calculates the gradients for the Z-values, the weights, and the biases
        Assumes initialize_params has been called.
        
        Parameters:
            - Y: the true labels, has shape (C, m), where C is the 
                number of classes, and m is the number of examples
        """
        m = Y.shape[1]
        A_L = self.As[-1]
        A_L_minus_one = self.As[-2]

        # handle output layer
        dZ_L = A_L - Y
        dW_L = 1./m * (dZ_L @ A_L_minus_one.T) + self.lambd/m*self.weights[-1]
        db_L = 1./m * np.sum(dZ_L, axis=1, keepdims=True)
        dA_prev = self.weights[-1].T @ dZ_L
        self.Zs_grads[-1] = dZ_L
        self.weights_grads[-1] = dW_L        
        self.biases_grads[-1] = db_L
        for i in range(len(self.layers)-2, 0, -1):
            dA_prev = self.back_prop_single_layer(dA_prev, i)

    def back_prop_single_layer(self, dA, num_layer):
        """
        Calculates the gradients for a single layer
        The gradients are stored in the NNModel object
        Assumes initialize_params has been called.
        
        Parameters:
            - dA: the gradients of the activations of the current layer
            - num_layer: which layer to calculate the gradients for

        Returns:
            - dA_prev: the gradients of the activations of the previous layer
        """
        m = dA.shape[1]
        Z = self.Zs[num_layer]
        dZ = dA * self.activation_grad(Z)
        dW = 1./m * (dZ @ self.As[num_layer-1].T) + \
            self.lambd/m*self.weights[num_layer]
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)        
        dA_prev = self.weights[num_layer].T @ dZ
        self.Zs_grads[num_layer] = dZ
        self.weights_grads[num_layer] = dW
        self.biases_grads[num_layer] = db
        return dA_prev

    def params_to_vec(self, weights, biases):
        """
        Turns the weights and biases of the model into one vector
        for gradient checking
        Assumes initialize_params has been called.
        
        Parametes:
            - weights: list of weights or the gradient of the weights
            - biases: list of biases or the gradient of the biases
        Returns:
            - vec: a numpy array of shape (1, params), where params is the
                total number of parameters in the model
        """
        vec = np.append(weights[1].reshape((1, -1)), 
                        biases[1].reshape((1, -1)))
        for i in range(2, len(self.layers)):
            vec = np.append(vec, 
                            weights[i].reshape((1, -1)))
            vec = np.append(vec,
                            biases[i].reshape((1, -1)))
        vec = vec.reshape((1, -1))
        return vec
        
    def vec_to_params(self, vec):
        """
        Turns an unrolled vector of weights and biases into a list of 
        weights and a list of baises in same format of NNModel's
        weights and biases
        
        Parameters:
            - vec: a vector of weights and baises in the same format
                as the output of params_to_vec(); is of shape (1, params),
                where params is the total number of parameters in the model
        
        Returns:
            - weights: a list of the weights that are in vec
            - biases: a list of the biases that are in vec
        """
        weights = [0]
        biases = [0]
        params_rolled = 0
        for i in range(1, len(self.layers)):
            num_weights = self.weights[i].shape[0] * self.weights[i].shape[1]
            W = vec[:, params_rolled:(params_rolled+num_weights)].reshape(
                self.weights[i].shape[0], self.weights[i].shape[1])
            weights.append(W)
            params_rolled += num_weights
            
            num_biases = self.biases[i].shape[0]            
            b = vec[:, params_rolled:(params_rolled+num_biases)].reshape(
                num_biases, 1)
            biases.append(b)
            params_rolled += num_biases
        return weights, biases
        
    def check_grads(self, X, Y, threshold, epsilon=10e-7):
        """
        Uses the 2-sided difference to compute the gradient
        This function is for checking the correctness of backprop.
        Assumes initialize_params has been called.
        
        Parameters:
            - X: input to use
            - Y: ground-truth-lables to use
            - threshold: how big a difference between dTheta and
                dTheta_approx is tolerated
            - epsilon: amount to add to each parameter
        """
        # compute dTheta
        self.forward_pass(X)
        self.backward_prop(Y)
        dTheta = self.params_to_vec(self.weights_grads, self.biases_grads)
        
        # compute dTheta_approx
        dTheta_approx = np.zeros(dTheta.shape)
        unrolled_params = self.params_to_vec(self.weights, self.biases)
        for i in range(unrolled_params.shape[1]):
            param_vec_plus_epsilon = unrolled_params.copy()
            param_vec_plus_epsilon[0, i] += epsilon
            
            param_vec_minus_epsilon = unrolled_params.copy()
            param_vec_minus_epsilon[0, i] -= epsilon
            
            weights_plus_epsilon, biases_plus_epsilon = \
                self.vec_to_params(param_vec_plus_epsilon)
            
            weights_minus_epsilon, biases_minus_epsilon = \
                self.vec_to_params(param_vec_minus_epsilon)
                
            nn_plus_epsilon = NNModel(self.layers, self.activation_func,
                                      self.activation_grad, self.lambd)
            nn_plus_epsilon.weights = weights_plus_epsilon
            nn_plus_epsilon.biases = biases_plus_epsilon
            
            nn_minus_epsilon = NNModel(self.layers, self.activation_func,
                                       self.activation_grad, self.lambd)
            nn_minus_epsilon.weights = weights_minus_epsilon
            nn_minus_epsilon.biases = biases_minus_epsilon
            
            plus_epsilon_A_L = nn_plus_epsilon.forward_pass(X)
            plus_epsilon_cost = nn_plus_epsilon.compute_cost_with_reglztn(Y, 
                plus_epsilon_A_L)
                
            minus_epsilon_A_L = nn_minus_epsilon.forward_pass(X)
            minus_epsilon_cost = nn_minus_epsilon.compute_cost_with_reglztn(Y,
                minus_epsilon_A_L)
            
            dTheta_approx[0, i] = \
                (plus_epsilon_cost - minus_epsilon_cost)/(2*epsilon)

        difference = np.linalg.norm(dTheta_approx - dTheta) / (
            np.linalg.norm(dTheta_approx) + np.linalg.norm(dTheta))
        if difference < threshold:
            print('Backprop and 2-sided difference are about equal')
        else:
            print('Backprop and 2-sided difference NOT equal',
                  'difference of:', difference)
            print(dTheta_approx - dTheta)
            weights_diffs, biases_diffs = \
                self.vec_to_params(dTheta_approx-dTheta)
            print('weights diffs:', weights_diffs)
            print('biases diffs:', biases_diffs)
            print('weights:', self.weights)
            print('biases:', self.biases)
            
    def update_parameters(self, learning_rate=0.01):
        """
        Implements gradient descent to update the weights and biases
        
        Parameters:
            - learning_rate: the learning rate to use in graidient descent            
        """        
        for i in range(1, len(self.layers)):
            self.weights[i] -= learning_rate * self.weights_grads[i]
            self.biases[i] -= learning_rate * self.biases_grads[i]
            
    def train(self, X, Y, iterations=10, learning_rate=0.01,
        verbose=0):
        """
        Runs several iterations consisting of:
            - a forward pass
            - a backward pass
            - updating the parameters once
            
        Parameters:
            - X: the input to the neural network
            - Y: the true labels, has shape (C, m), where C is the 
                number of classes, and m is the number of examples
            - iterations: the number of iterations to run
            - learning_rate: the learning rate to use for gradient descent
            - verbose: set to 0 for no messages, 1 to see the costs
                every 50 iterations
        """
        NUM_ITERATIONS = 50
        
        for i in range(iterations):
            A_L = self.forward_pass(X)
            self.backward_prop(Y)
            self.update_parameters(learning_rate)
            if i % NUM_ITERATIONS == 0 and verbose == 1:
                cost = self.compute_cost(Y, A_L)
                print('Iteration %s: Cost = %.2f' % (i, cost))
        return
        
    def predict(self, X):
        """
        Returns the predictions for a matrix of data points
        
        Parameters:
            - X: the data points to predict the labels for
            
        Returns:
            - Y: the predicted labels
        """
        temp_nn = NNModel(self.layers, self.activation_func, 
                      self.activation_grad, self.lambd)
        temp_nn.weights = self.weights
        temp_nn.biases = self.biases
        A_L = temp_nn.forward_pass(X)
        Y = np.argmax(A_L, axis=0)
        return Y
        
    def get_mini_batches(self, X, Y, mini_batch_size):
        """
        Returns a list of mini-batches of X and Y
        
        Parameters:
            - X: the input to the neural network
            - Y: the true labels, has shape (C, m), where C is the 
                number of classes, and m is the number of examples
            - mini_batch_size: the size of each mini-batch
            
        Returns:
            - mini_batches_X: a list of mini-batches of X 
            - mini_batches_Y: a list of mini-batches of Y
        """
        m = X.shape[1]
        mini_batches_X = []
        mini_batches_Y = []
        rand_indices = np.random.permutation(m)
        shuffled_X = X[:, rand_indices]
        shuffled_Y = Y[:, rand_indices]
        count = 0
        for i in range(m // mini_batch_size):
            mini_batches_X.append(shuffled_X[:,count:(count+mini_batch_size)])
            mini_batches_Y.append(shuffled_Y[:,count:(count+mini_batch_size)])
            count += mini_batch_size
        if m % mini_batch_size != 0:
            mini_batches_X.append(shuffled_X[:,count:])
            mini_batches_Y.append(shuffled_Y[:,count:])
        return mini_batches_X, mini_batches_Y
        
    def train_mini_batch(self, X, Y, epochs, mini_batch_size, 
                         learning_rate=0.01, show_cost_every_num_iter=None):
        """
        Runs mini-batch gradient descent
        
        Parameters:
            - X: the input to the neural network
            - Y: the true labels, has shape (C, m), where C is the 
                number of classes, and m is the number of examples
            - epochs: the number of times to go through the training set X
            - mini_batch_size: the size of each mini-batch
            - learning_rate: the learning rate to use for gradient descent
            - show_cost_every_num_iter: if an integer, prints the overall cost
                after every show_cost_every_num_iter of iterations
        """
        iter = 0
        
        for i in range(epochs):
            mini_batches_X, mini_batches_Y = self.get_mini_batches(X, Y,
                mini_batch_size)
            for j in range(len(mini_batches_X)):
                self.train(mini_batches_X[j], mini_batches_Y[j],
                           1, learning_rate)
                if show_cost_every_num_iter is not None and \
                    iter % show_cost_every_num_iter == 0:
                    A_L = self.forward_pass(X)
                    cost = self.compute_cost(Y, A_L)
                    print('Iteration: %s, Cost: %.4f' % (iter, cost))
                iter += 1
        
def relu(X):
    """
    Implements the rectified linear unit function
    
    Parameters:
        - X: a numpy array
        
    Returns:
        - a numpy array with relu applied element-wise to x
    """
    
    mask = X > np.zeros(X.shape)
    return np.abs(X * mask)
    
def relu_grad(X):
    """
    Calculates the gradient of the rectified linear unit function

    Parameters:
        - X: a numpy array

    Returns:
        - a numpy array with the gradient of the relu calculated element-wise
    """
    return (X > 0) * 1.
   
