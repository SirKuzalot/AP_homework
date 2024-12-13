#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001,**kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            self.W[y_i, :] += learning_rate * x_i
            self.W[y_hat, :] -= learning_rate * x_i


        #raise NotImplementedError # Q1.1 (a)


class LogisticRegression(LinearModel):

    def softmax(self, z):
        exps = np.exp(z - np.max(z))  # Stability improvement
        return exps / np.sum(exps)
    
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """

        # Compute the linear scores
        scores = self.W.dot(x_i)  # Shape: (n_classes,)
        
        # Compute probabilities using softmax
        probs = self.softmax(scores)  # Shape: (n_classes,)
        
        # Create one-hot encoding for the true label
        y_one_hot = np.zeros_like(probs)
        y_one_hot[y_i] = 1
        
        # Compute gradient of the cross-entropy loss
        gradient = np.outer(probs - y_one_hot, x_i)  # Shape: (n_classes, n_features)
        
        # Update the weights
        self.W -= learning_rate * (gradient + l2_penalty * self.W)


        #raise NotImplementedError # Q1.2 (a,b)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        """
        Initialize an MLP with a single hidden layer.
        
        Args:
            n_classes (int): Number of output classes.
            n_features (int): Number of input features.
            hidden_size (int): Number of hidden units.
        """
        # Initialize weights and biases
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size,))
        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros((n_classes,))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(n_samples), y_true] + 1*(10**-6) )
        return np.sum(log_likelihood) / n_samples
    
    def predict(self, X):
        """Compute the forward pass for prediction."""
        hidden = self.relu(np.dot(X, self.W1.T) + self.b1)
        output = self.softmax(np.dot(hidden, self.W2.T) + self.b2)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Train the model for one epoch using stochastic gradient descent.

        Args:
            X (ndarray): Input data of shape (n_samples, n_features).
            y (ndarray): True labels of shape (n_samples,).
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            float: The loss for the epoch.
        """
        n_samples = X.shape[0]
        loss = 0

        for i in range(n_samples):
            # Forward pass
            xi = X[i:i+1]  # Single sample
            yi = y[i]
            
            hidden = self.relu(np.dot(xi, self.W1.T) + self.b1)
            output = self.softmax(np.dot(hidden, self.W2.T) + self.b2)
            
            # Compute loss using the defined loss function
            loss += self.cross_entropy_loss(np.array([yi]), output)

            # Backward pass
            grad_output = output
            grad_output[0, yi] -= 1

            grad_W2 = np.dot(grad_output.T, hidden)
            grad_b2 = grad_output.flatten()
            
            grad_hidden = np.dot(grad_output, self.W2) * self.relu_derivative(hidden)
            grad_W1 = np.dot(grad_hidden.T, xi)
            grad_b1 = grad_hidden.flatten()

            # Update weights and biases
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2
            self.W1 -= learning_rate * grad_W1
            self.b1 -= learning_rate * grad_b1

        return loss / n_samples


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
