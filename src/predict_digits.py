# predict_digits.py
# Daniel Lin, January 2019
# This script trains a neural network on a subset of the MNIST dataset

import nn
import load_training_data
import numpy as np
import pandas as pd

TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
NUM_EPOCHS = 10
INPUT_SIZE = 784
OUTPUT_SIZE = 10

def get_acc(nn, X, Y):
    """
    Calculates the accuracy of a neural network 
    
    Parameters:
        - nn: the neural network
        - X: the design matrix with shape (n, m), where n is the number
            of features and m is the number of examples
        - Y: the labels as an array with shape (m,), where m is the
            number of examples
        
    Returns:
        - acc: the accuracy nn gets on predicting the labels
    """
    m = X.shape[1]
    Y_hat = nn.predict(X)
    acc = np.sum(Y==Y_hat)/m
    return acc
    
def search_hyperparams(alpha_range, mbs_range, hid_units_range, layers_range,
                       lambda_range, input_size, 
                       output_size, num_hp_samples=10):
    """
    Randomly tries different values for hyperparameters
    
    Parameters:
        - alpha_range: a sequence with two elements, (a, b)
            the learning rate alpha will range over 10**a to 10**b
        - mbs_range: a sequence with two elements, (a, b)
            the mini-batch size will range from 2**a to 2**b
        - hid_units_range: a sequence with two elements, (a, b)
            the number of hidden units will range from a to b
        - layers_range: a sequence with two elements, (a, b)
            the number of hidden layers will range from a to b
        - lambda_range: a sequence with two elements, (a, b)
            the regularization hyperparameter lambda will range from a to b
        - input_size: the size of the input layer
        - output_size: the size of the output layer
        - num_hp_samples: the number of times to sample from the 
            space of hyperparameters
    
    Returns:
        - models_info: a list of tuples, where each tuple is:
            (training accuracy, validation accuracy, alpha, mini-batch size,
            number of hidden units, number of hidden layers, lambda)
            The list is sorted descendingly by validation accuracy
    """
    models_info = []
    for i in range(num_hp_samples):
        print('Model %s out of %s' % ((i+1), num_hp_samples))
        # learning rate
        r = np.random.rand() * (alpha_range[1] - alpha_range[0]) \
            + alpha_range[0]
        alpha = 10 ** r
        
        # mini-batch size
        s = np.random.randint(mbs_range[0], mbs_range[1])
        mbs = 2 ** s
        
        # number of hidden units
        num_hidden_units = np.random.randint(hid_units_range[0], 
                                             hid_units_range[1])
        
        # number of hidden layers
        num_hidden_layers = np.random.randint(layers_range[0], 
                                              layers_range[1])
        
        # regularization hyperparameter, lambda
        lambd = np.random.rand() * (lambda_range[1] - lambda_range[0]) \
            + lambda_range[0]
        
        ntwk_arch = [input_size] + [num_hidden_units] * num_hidden_layers \
            + [output_size]
        network = nn.NNModel(ntwk_arch, nn.relu, nn.relu_grad, 
                             lambd=lambd)
        network.train_mini_batch(X_train, Y_train_oh, epochs=NUM_EPOCHS, 
                                 mini_batch_size=mbs, learning_rate=alpha,
                                 show_cost_every_num_iter=None)
        train_acc = get_acc(network, X_train, Y_train)
        val_acc = get_acc(network, X_val, Y_val)
        
        # add to list of models tried in format:
        # (training accuracy, validation accuracy, alpha, mini-batch size,
        # number of hidden units, number of hidden layers, lambda)
        model_info = (train_acc, val_acc, alpha, mbs, num_hidden_units,
                           num_hidden_layers, lambd)
        models_info.append(model_info)

    # print out the networks based on which had the best validation accuracy
    models_info.sort(reverse=True, key=lambda model: model[1])
    return models_info   
    
X, Y = load_training_data.read_data('data/train.csv')
Y_oh = pd.get_dummies(Y.flatten()).values.T
m = X.shape[1]
train_num = int(m * TRAIN_SIZE)
val_num = int(m * VAL_SIZE)
test_num = m - train_num - val_num
X_train = X[:, :train_num]
Y_train_oh = Y_oh[:, :train_num]
Y_train = Y[:, :train_num]
X_val = X[:, train_num:(train_num+val_num)]
Y_val_oh = Y_oh[:, train_num:(train_num+val_num)]
Y_val = Y[:, train_num:(train_num+val_num)]
X_test = X[:, (train_num+val_num):]
Y_test_oh = Y_oh[:, (train_num+val_num):]
Y_test = Y[:, (train_num+val_num):]

# hyperparameter tuning
first_search = search_hyperparams((-4, 0), (1, 10), (50, 200), (2, 5),
                       (1, 10), INPUT_SIZE, 
                       OUTPUT_SIZE, num_hp_samples=10)
print(first_search)                       
second_search = search_hyperparams((-2, 0), (5, 10), (85, 200), (3, 5),
                       (1, 5), INPUT_SIZE, 
                       OUTPUT_SIZE, num_hp_samples=10)
print(second_search) 
third_search = search_hyperparams((-2, 0), (5, 10), (95, 200), (3, 7),
                       (1, 4), INPUT_SIZE, 
                       OUTPUT_SIZE, num_hp_samples=10)
print(third_search) 
fourth_search = search_hyperparams((-2, 0), (6, 10), (95, 150), (3, 7),
                       (1, 4), INPUT_SIZE, 
                       OUTPUT_SIZE, num_hp_samples=10)
print(fourth_search) 

network_1 = nn.NNModel([INPUT_SIZE, 225, 225, 225, 225, 225, OUTPUT_SIZE], 
                        nn.relu, nn.relu_grad, lambd=1.263)
network_1.train_mini_batch(X_train, Y_train_oh, epochs=15, 
                           mini_batch_size=256, learning_rate=.0715,
                           show_cost_every_num_iter=None)
train_acc = get_acc(network_1, X_train, Y_train)
val_acc = get_acc(network_1, X_val, Y_val)
test_acc = get_acc(network_1, X_test, Y_test)
print('Training Accuracy %.4f' % train_acc)
print('Validation Accuracy %.4f' % val_acc) 
print('Test Accuracy %.4f' % test_acc)                          