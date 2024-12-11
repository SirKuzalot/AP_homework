import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train models for Intel Landscape Classification")
    parser.add_argument('model', choices=['perceptron', 'logistic_regression'],
                        help="Model to train: perceptron or logistic_regression")
    parser.add_argument('-epochs', type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument('-l2_penalty', type=float, default=0.0,
                    help="Strength of L2 regularization (default: 0.0)")
    return parser.parse_args()

class Perceptron:
    def __init__(self, num_classes, input_dim, eta=1):
        self.W = np.zeros((num_classes, input_dim)) 
        self.eta = eta

    def update_weights(self, X, y):
        mistakes = 0
        for x, label in zip(X, y):
            y_hat = np.argmax(self.W.dot(x))
            if y_hat != label:
                mistakes += 1
                self.W[label, :] += self.eta * x
                self.W[y_hat, :] -= self.eta * x
        return mistakes

    def predict(self, X):
        return np.argmax(self.W.dot(X.T), axis=0)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
def train_perceptron(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100):
    num_classes = 6
    input_dim = X_train.shape[1]
    perceptron = Perceptron(num_classes, input_dim)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        mistakes = perceptron.update_weights(X_train, y_train)
        train_acc = perceptron.evaluate(X_train, y_train)
        val_acc = perceptron.evaluate(X_val, y_val)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}: Mistakes: {mistakes}, "
              f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_acc = perceptron.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    return train_accuracies, val_accuracies
class LogisticRegression:
    def __init__(self, num_classes, input_dim, eta=0.001, l2_penalty=0.0):
        self.W = np.zeros((num_classes, input_dim))  # Weight matrix
        self.eta = eta
        self.l2_penalty = l2_penalty  # L2 regularization strength

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exps = np.exp(z - np.max(z))  # Stability improvement
        return exps / np.sum(exps)

    def update_weights(self, X, y):
        mistakes = 0
        for x, label in zip(X, y):
            scores = self.W.dot(x)  # Linear scores for each class
            probs = self.softmax(scores)  # Softmax for multi-class probabilities
            y_hat = np.argmax(probs)
            if y_hat != label:
                mistakes += 1

            # Gradient calculation
            gradient = probs
            gradient[label] -= 1  # One-hot encoding difference
            
            # Update weights with L2 regularization
            self.W -= self.eta * (np.outer(gradient, x) + self.l2_penalty * self.W)
        
        return mistakes

    def predict(self, X):
        scores = self.W.dot(X.T)
        return np.argmax(scores, axis=0)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, eta=0.001, l2_penalty=0.0):
    num_classes = 6
    input_dim = X_train.shape[1]
    model = LogisticRegression(num_classes, input_dim, eta, l2_penalty)

    train_accuracies = []
    val_accuracies = []
    weight_norms = []  # Store Frobenius norms

    for epoch in range(epochs):
        mistakes = model.update_weights(X_train, y_train)
        train_acc = model.evaluate(X_train, y_train)
        val_acc = model.evaluate(X_val, y_val)

        # Compute and store the Frobenius norm of weights
        frobenius_norm = np.sqrt(np.sum(np.square(model.W)))
        weight_norms.append(frobenius_norm)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}: Mistakes: {mistakes}, "
              f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}, "
              f"Weights Norm: {frobenius_norm:.4f}")

    test_acc = model.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    return train_accuracies, val_accuracies, weight_norms




def plot_accuracies(train_accuracies, val_accuracies):
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Perceptron Performance')
    plt.savefig('plot.png')
    
def plot_weight_norms(weight_norms_non_regularized, weight_norms_regularized):
    plt.plot(weight_norms_non_regularized, label='Non-Regularized')
    plt.plot(weight_norms_regularized, label='Regularized')
    plt.xlabel('Epochs')
    plt.ylabel('Frobenius Norm of Weights')
    plt.legend()
    plt.title('Impact of Regularization on Weight Norms')
    plt.savefig('weight_norms.png')
    
    
if __name__ == "__main__":
    args = parse_arguments()
    
    # Load dataset
    data = np.load('intel_landscapes.v2.npz')
    X_train, y_train = data['train_images'], data['train_labels']
    X_val, y_val = data['val_images'], data['val_labels']
    X_test, y_test = data['test_images'], data['test_labels']
    
    # Flatten images for linear classifiers
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    if args.model == 'perceptron':
        print(f"Training Perceptron for {args.epochs} epochs...")
        train_accuracies, val_accuracies = train_perceptron(
            X_train, y_train, X_val, y_val, X_test, y_test, args.epochs
        )
        plot_accuracies(train_accuracies, val_accuracies)
    
    elif args.model == 'logistic_regression':
        train_accuracies, val_accuracies, weight_norms = train_logistic_regression(
            X_train, y_train, X_val, y_val, X_test, y_test, args.epochs, l2_penalty=args.l2_penalty
        )

        plot_accuracies(train_accuracies, val_accuracies)

# Fica a faltar a ultima parte. Ainda n revi, faço isso amanha. Podes rever se quiseres ou podes so tentar o 2

