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
        if x.ndim == 1:  # Handle 1D input
            exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
            return exp_x / np.sum(exp_x)
        elif x.ndim == 2:  # Handle 2D input
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError("Input to softmax must be 1D or 2D.")


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
            float: The average loss for the epoch.
        """
        n_samples = X.shape[0]
        total_loss = 0

        for i in range(n_samples):
            # Extract a single sample and its label
            xi = X[i]  # Shape: (n_features,)
            yi = y[i]  # True class index

            ### FORWARD PASS ###
            # First layer: Input to hidden
            z1 = self.W1.dot(xi) + self.b1  # Pre-activation: Shape (hidden_size,)
            h1 = self.relu(z1)                  # Activation (ReLU): Shape (hidden_size,)

            # Second layer: Hidden to output
            z2 = self.W2.dot(h1) + self.b2  # Pre-activation: Shape (n_classes,)
            y_pred = self.softmax(z2)           # Output probabilities: Shape (n_classes,)

            # Compute cross-entropy loss for this sample
            loss = -np.log(y_pred[yi] + 10**-6)
            total_loss += loss

            ### BACKWARD PASS ###
            # Gradients for the output layer
            z2_grad = y_pred  # Gradient of loss w.r.t. softmax inputs
            z2_grad[yi] -= 1  # Adjust for the true class index

            w2_grad = z2_grad[:, None].dot(h1[:, None].T)  # Weight gradients (outer product)
            b2_grad = z2_grad                # Bias gradients

            # Gradients for the hidden layer
            h1_grad = self.W2.T.dot(z2_grad)  # Backpropagated gradient to hidden
            z1_grad = h1_grad * self.relu_derivative(z1)  # Apply ReLU derivative

            w1_grad = z1_grad[:, None].dot(xi[:, None].T)  # Weight gradients (outer product)
            b1_grad = z1_grad                # Bias gradients

            ### UPDATE PARAMETERS ###
            self.W2 -= learning_rate * w2_grad
            self.b2 -= learning_rate * b2_grad
            self.W1 -= learning_rate * w1_grad
            self.b1 -= learning_rate * b1_grad

        # Return the average loss over all samples
        return total_loss / n_samples
    
'''
initial train acc: 0.1694 | initial val acc: 0.1880
Training epoch 1
loss: 4.0943 | train acc: 0.3845 | val acc: 0.3604
Training epoch 2
loss: 1.6373 | train acc: 0.4567 | val acc: 0.4352
Training epoch 3
loss: 1.4623 | train acc: 0.4808 | val acc: 0.4551
Training epoch 4
loss: 1.3732 | train acc: 0.4743 | val acc: 0.4366
Training epoch 5
loss: 1.3285 | train acc: 0.4667 | val acc: 0.4309
Training epoch 6
loss: 1.3021 | train acc: 0.5260 | val acc: 0.4822
Training epoch 7
loss: 1.2796 | train acc: 0.5226 | val acc: 0.4793
Training epoch 8
loss: 1.2653 | train acc: 0.4929 | val acc: 0.4615
Training epoch 9
loss: 1.2566 | train acc: 0.5257 | val acc: 0.4793
Training epoch 10
loss: 1.2445 | train acc: 0.5421 | val acc: 0.5014
Training epoch 11
loss: 1.2341 | train acc: 0.5264 | val acc: 0.4900
Training epoch 12
loss: 1.2175 | train acc: 0.4849 | val acc: 0.4402
Training epoch 13
loss: 1.2065 | train acc: 0.5184 | val acc: 0.4729
Training epoch 14
loss: 1.1984 | train acc: 0.5410 | val acc: 0.4815
Training epoch 15
loss: 1.1909 | train acc: 0.5544 | val acc: 0.5100
Training epoch 16
loss: 1.1867 | train acc: 0.5523 | val acc: 0.4865
Training epoch 17
loss: 1.1811 | train acc: 0.5668 | val acc: 0.5107
Training epoch 18
loss: 1.1722 | train acc: 0.5648 | val acc: 0.5050
Training epoch 19
loss: 1.1641 | train acc: 0.5690 | val acc: 0.5043
Training epoch 20
loss: 1.1493 | train acc: 0.5835 | val acc: 0.5256
Training took 23 minutes and 8 seconds
Final test acc: 0.5307
'''



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
