# Using Neural Networks for Recognizing Handwritten Digits

This repository includes files for recognizing handwritten digits, which come from the MNIST dataset.  The file load_training_data.py reads in the data, which is stored in a csv file.  The script also preprocesses the data.

The file nn.py implements a neural network using Python and numpy.  The class `NNModel` creates a neural network object that has a softmax activation for the output layer.  Included in nn.py is a function for ReLU and a function for the ReLU gradient (`relu` and `relu_grad`).  These two functions can be passed to the `NNModel` constructor.

The file predict_digits.py uses load_training_data.py and nn.py to categorize handwritten digits.  This file includes the hyperparameter search.

## Results

Using a neural network with 5 hidden layers, each with 225 hidden units resulted in a test accuracy of 95%.

## Hyperparameter Search

I performed a random search over the following hyperparameters:

- Learning rate, $\alpha$
- Mini-batch size
- Size of each hidden layer
- Number of hidden layers
- Regularization hyperparameter, $\lambda$

Based on the random search, I settled on the following hyperparameter values:

| Hyperparameter | Value |
| -------------- | ----- |
| Learning Rate, $\alpha$| 0.0715 |
| Mini-Batch Size | 256 |
| Hidden Layer Size | 225 |
| Number of Hidden Layers | 5 |
| Regularization Hyperparameter, $\lambda$ | 1.263 |

## Using the `NNModel` Class to Build a Neural Network

### Initialization
```
network = NNModel([100, 5, 10], relu, relu_grad, lambd=0)
```
The first argument is a list of the layer sizes, with the input layer first and the output layer last.  Here, we have a network with an input layer of size 100, a hidden layer with 5 hidden units, and an output layer with 10 units.

The second argument is the activation function, which is ReLU in this example.

The third argument is the gradient of the activation function.

The fourth argument is the regularization hyperparameter, `lambd`.  A bigger value means more regularization.

The network automatically initializes the weights randomly using He initialization.

### Training
After creating a `NNModel` object, you can train using the `train` method for batch gradient descent:

```
network.train(X, Y, iterations=700, learning_rate=0.001, verbose=1)
```
In the above example, `X` is the design matrix, and `Y` is a one-hot matrix containing the labels.

`iterations=700` will make the network train for 700 iterations.

`learning_rate=0.001` sets the learning rate to 0.001.

`verbose=1` will cause the network to print out the cost every 50 iterations.  Setting `verbose` to 0 will make the cost not be printed at all.

Instead of batch gradient descent, you can use mini-batch gradient descent with `train_mini_batch`:

```
network.train_mini_batch(X, Y, epochs=10, mini_batch_size=128, learning_rate=0.1, show_cost_every_num_iter=50)
```

Again, `X` and `Y` are the design and label matrices respectively.

`epochs=10` sets the number of epochs to 10.

`mini_batch_size=128` uses a mini-batch size of 128.

`learning_rate=0.1` sets the learning rate to 0.1.

`show_cost_every_num_iter=50` causes the model to print out the cost every 50 iterations.

### Predicting
To predict the labels given some design matrix, use the `predict` method:

```
network.predict(X_new)
```

`X_new` is the design matrix that we want to predict the labels for.