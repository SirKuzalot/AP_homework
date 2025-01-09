#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

import utils


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, maxpool=True, batch_norm=True, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, dropout_prob=0.1, maxpool=True, batch_norm=True):
        super().__init__()
        channels = [3, 32, 64, 128]
        fc1_out_dim = 1024
        fc2_out_dim = 512

        self.conv_blocks = nn.Sequential(
            ConvBlock(channels[0], channels[1], maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob),
            ConvBlock(channels[1], channels[2], maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob),
            ConvBlock(channels[2], channels[3], maxpool=maxpool, batch_norm=batch_norm, dropout=dropout_prob)
        )

        # Use adaptive pooling to ensure the output shape after convolutions is consistent
        self.globalAveragePool = nn.AdaptiveAvgPool2d((1, 1)) if batch_norm else nn.Identity()

        # Calculate the output size after the convolutional layers and pooling
        self.fc1 = nn.Linear(channels[3], fc1_out_dim) if batch_norm else nn.Linear(channels[3]*6*6, fc1_out_dim)
        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.fc3 = nn.Linear(fc2_out_dim, 6)

        self.fc1_bn = nn.BatchNorm1d(fc1_out_dim) if batch_norm else nn.Identity()
        self.fc2_bn = nn.BatchNorm1d(fc2_out_dim) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 48, 48)
        x = self.conv_blocks(x)
        
        # Apply global average pooling
        x = self.globalAveragePool(x)
        
        # Flatten the tensor
        x = torch.flatten(x, 1)  # This will now give (batch_size, channels[3])
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

 

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels


def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    """
    Returns the number of trainable parameters in the given model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_batch_norm', action='store_true')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    parser.add_argument('-device', choices=['cpu', 'cuda', 'mps'], default='cpu')

    opt = parser.parse_args()

    # Setting seed for reproducibility
    utils.configure_seed(seed=42)

    # Load data
    data = utils.load_dataset(data_path=opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)
    test_X, test_y = dataset.test_X.to(opt.device), dataset.test_y.to(opt.device)

    # initialize the model
    model = CNN(
        opt.dropout,
        maxpool=not opt.no_maxpool,
        batch_norm=not opt.no_batch_norm
    ).to(opt.device)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.NLLLoss()

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('\nTraining epoch {}'.format(ii))
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(opt.device)
            y_batch = y_batch.to(opt.device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_acc, val_loss = evaluate(model, dev_X, dev_y, criterion)
        valid_accs.append(val_acc)
        print("Valid loss: %.4f" % val_loss)
        print('Valid acc: %.4f' % val_acc)

    test_acc, _ = evaluate(model, test_X, test_y, criterion)
    test_acc_perc = test_acc * 100
    test_acc_str = '%.2f' % test_acc_perc
    print('Final Test acc: %.4f' % test_acc)
    # plot
    sufix = plot_file_name_sufix(opt, exlude={'data_path', 'device'})

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-3-train-loss-{}-{}'.format(sufix, test_acc_str))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-3-valid-accuracy-{}-{}'.format(sufix, test_acc_str))

    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
#    
#    Training epoch 1
#Training loss: 1.7270
#Valid loss: 1.5440
#Valid acc: 0.3654
#
#Training epoch 2
#Training loss: 1.5835
#Valid loss: 1.3135
#Valid acc: 0.4950
#
#Training epoch 3
#Training loss: 1.4868
#Valid loss: 1.3270
#Valid acc: 0.4644
#
#Training epoch 4
#Training loss: 1.4259
#Valid loss: 1.3403
#Valid acc: 0.4751
#
#Training epoch 5
#Training loss: 1.3789
#Valid loss: 1.1573
#Valid acc: 0.5442
#
#Training epoch 6
#Training loss: 1.3348
#Valid loss: 1.1085
#Valid acc: 0.5570
#
#Training epoch 7
#Training loss: 1.2952
#Valid loss: 1.0301
#Valid acc: 0.6068
#
#Training epoch 8
#Training loss: 1.2596
#Valid loss: 0.9456
#Valid acc: 0.6460
#
#Training epoch 9
#Training loss: 1.2276
#Valid loss: 0.9880
#Valid acc: 0.6083
#
#Training epoch 10
#Training loss: 1.1978
#Valid loss: 0.9729
#Valid acc: 0.6332
#
#Training epoch 11
#Training loss: 1.1697
#Valid loss: 0.9642
#Valid acc: 0.6396
#
#Training epoch 12
#Training loss: 1.1443
#Valid loss: 1.0021
#Valid acc: 0.6182
#
#Training epoch 13
#Training loss: 1.1195
#Valid loss: 0.8943
#Valid acc: 0.6624
#
#Training epoch 14
#Training loss: 1.0972
#Valid loss: 0.8796
#Valid acc: 0.6838
#
#Training epoch 15
#Training loss: 1.0757
#Valid loss: 0.9061
#Valid acc: 0.6567
#
#Training epoch 16
#Training loss: 1.0556
#Valid loss: 0.8497
#Valid acc: 0.6781
#
#Training epoch 17
#Training loss: 1.0362
#Valid loss: 0.7978
#Valid acc: 0.7087
#
#Training epoch 18
#Training loss: 1.0173
#Valid loss: 0.7728
#Valid acc: 0.7066
#
#Training epoch 19
#Training loss: 0.9992
#Valid loss: 0.8423
#Valid acc: 0.6923
#
#Training epoch 20
#Training loss: 0.9818
#Valid loss: 0.8922
#Valid acc: 0.6845
#
#Training epoch 21
#Training loss: 0.9647
#Valid loss: 0.7761
#Valid acc: 0.7201
#
#Training epoch 22
#Training loss: 0.9480
#Valid loss: 0.8445
#Valid acc: 0.6980
#
#Training epoch 23
#Training loss: 0.9308
#Valid loss: 0.9066
#Valid acc: 0.6795
#
#Training epoch 24
#Training loss: 0.9139
#Valid loss: 0.8715
#Valid acc: 0.7080
#
#Training epoch 25
#Training loss: 0.8973
#Valid loss: 0.8518
#Valid acc: 0.6895
#
#Training epoch 26
#Training loss: 0.8806
#Valid loss: 0.8410
#Valid acc: 0.7101
#
#Training epoch 27
#Training loss: 0.8638
#Valid loss: 0.9263
#Valid acc: 0.6987
#
#Training epoch 28
#Training loss: 0.8471
#Valid loss: 0.9071
#Valid acc: 0.7115
#
#Training epoch 29
#Training loss: 0.8302
#Valid loss: 0.9174
#Valid acc: 0.6930
#
#Training epoch 30
#Training loss: 0.8133
#Valid loss: 0.9951
#Valid acc: 0.7094
#
#Training epoch 31
#Training loss: 0.7965
#Valid loss: 1.0991
#Valid acc: 0.6873
#
#Training epoch 32
#Training loss: 0.7802
#Valid loss: 1.0345
#Valid acc: 0.6909
#
#Training epoch 33
#Training loss: 0.7640
#Valid loss: 1.0206
#Valid acc: 0.7151
#
#Training epoch 34
#Training loss: 0.7477
#Valid loss: 1.1283
#Valid acc: 0.6895
#
#Training epoch 35
#Training loss: 0.7320
#Valid loss: 1.2149
#Valid acc: 0.7130
#
#Training epoch 36
#Training loss: 0.7167
#Valid loss: 1.1266
#Valid acc: 0.7244
#
#Training epoch 37
#Training loss: 0.7017
#Valid loss: 1.2683
#Valid acc: 0.6959
#
#Training epoch 38
#Training loss: 0.6871
#Valid loss: 1.2898
#Valid acc: 0.7044
#
#Training epoch 39
#Training loss: 0.6729
#Valid loss: 1.4617
#Valid acc: 0.6859
#
#Training epoch 40
#Training loss: 0.6591
#Valid loss: 1.2404
#Valid acc: 0.7144
#Final Test acc: 0.7113

# Number of trainable parameters without normalization:  5340742


#Training epoch 1
#Training loss: 1.3203
#Valid loss: 1.2479
#Valid acc: 0.5150
#
#Training epoch 2
#Training loss: 1.2280
#Valid loss: 1.1394
#Valid acc: 0.5627
#
#Training epoch 3
#Training loss: 1.1710
#Valid loss: 1.2559
#Valid acc: 0.5021
#
#Training epoch 4
#Training loss: 1.1286
#Valid loss: 1.2339
#Valid acc: 0.5605
#
#Training epoch 5
#Training loss: 1.0941
#Valid loss: 0.9579
#Valid acc: 0.6254
#
#Training epoch 6
#Training loss: 1.0678
#Valid loss: 0.8870
#Valid acc: 0.6795
#
#Training epoch 7
#Training loss: 1.0438
#Valid loss: 0.9209
#Valid acc: 0.6489
#
#Training epoch 8
#Training loss: 1.0234
#Valid loss: 0.8882
#Valid acc: 0.6930
#
#Training epoch 9
#Training loss: 1.0049
#Valid loss: 1.0156
#Valid acc: 0.6289
#
#Training epoch 10
#Training loss: 0.9869
#Valid loss: 0.7527
#Valid acc: 0.7244
#
#Training epoch 11
#Training loss: 0.9722
#Valid loss: 0.8835
#Valid acc: 0.6823
#
#Training epoch 12
#Training loss: 0.9580
#Valid loss: 0.8890
#Valid acc: 0.6802
#
#Training epoch 13
#Training loss: 0.9453
#Valid loss: 0.7678
#Valid acc: 0.7236
#
#Training epoch 14
#Training loss: 0.9327
#Valid loss: 0.7380
#Valid acc: 0.7336
#
#Training epoch 15
#Training loss: 0.9214
#Valid loss: 1.0170
#Valid acc: 0.6531
#
#Training epoch 16
#Training loss: 0.9102
#Valid loss: 0.8043
#Valid acc: 0.7137
#
#Training epoch 17
#Training loss: 0.8998
#Valid loss: 0.8547
#Valid acc: 0.7023
#
#Training epoch 18
#Training loss: 0.8900
#Valid loss: 0.7461
#Valid acc: 0.7322
#
#Training epoch 19
#Training loss: 0.8805
#Valid loss: 0.8767
#Valid acc: 0.7058
#
#Training epoch 20
#Training loss: 0.8713
#Valid loss: 0.7584
#Valid acc: 0.7293
#
#Training epoch 21
#Training loss: 0.8630
#Valid loss: 1.0221
#Valid acc: 0.6453
#
#Training epoch 22
#Training loss: 0.8546
#Valid loss: 0.7517
#Valid acc: 0.7379
#
#Training epoch 23
#Training loss: 0.8470
#Valid loss: 0.7431
#Valid acc: 0.7343
#
#Training epoch 24
#Training loss: 0.8395
#Valid loss: 0.7294
#Valid acc: 0.7272
#
#Training epoch 25
#Training loss: 0.8322
#Valid loss: 0.7904
#Valid acc: 0.7073
#
#Training epoch 26
#Training loss: 0.8248
#Valid loss: 0.7415
#Valid acc: 0.7286
#
#Training epoch 27
#Training loss: 0.8178
#Valid loss: 0.7177
#Valid acc: 0.7521
#
#Training epoch 28
#Training loss: 0.8111
#Valid loss: 0.8292
#Valid acc: 0.7037
#
#Training epoch 29
#Training loss: 0.8044
#Valid loss: 0.8475
#Valid acc: 0.7115
#
#Training epoch 30
#Training loss: 0.7983
#Valid loss: 0.7272
#Valid acc: 0.7415
#
#Training epoch 31
#Training loss: 0.7920
#Valid loss: 0.6882
#Valid acc: 0.7479
#
#Training epoch 32
#Training loss: 0.7861
#Valid loss: 0.8113
#Valid acc: 0.7422
#
#Training epoch 33
#Training loss: 0.7803
#Valid loss: 0.6591
#Valid acc: 0.7635
#
#Training epoch 34
#Training loss: 0.7745
#Valid loss: 0.8583
#Valid acc: 0.7030
#
#Training epoch 35
#Training loss: 0.7689
#Valid loss: 0.6802
#Valid acc: 0.7528
#
#Training epoch 36
#Training loss: 0.7637
#Valid loss: 0.7729
#Valid acc: 0.7258
#
#Training epoch 37
#Training loss: 0.7583
#Valid loss: 0.7832
#Valid acc: 0.7222
#
#Training epoch 38
#Training loss: 0.7529
#Valid loss: 0.7514
#Valid acc: 0.7486
#
#Training epoch 39
#Training loss: 0.7475
#Valid loss: 0.7251
#Valid acc: 0.7400
#
#Training epoch 40
#Training loss: 0.7424
#Valid loss: 0.6923
#Valid acc: 0.7528
#Final Test acc: 0.7517
    
#Number of trainable parameters with normalization:  756742

