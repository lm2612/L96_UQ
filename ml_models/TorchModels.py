import numpy as np
import torch

class LinearRegression(torch.nn.Module):
    """Linear regression"""
    def __init__(self, n_features=1, n_targets=1):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_targets)
    
    def forward(self, X):
        return self.linear(X)

class NN(torch.nn.Module):
    """Simple feedforward neural network with two hidden layers"""
    def __init__(self, n_features=1, n_targets=1, n_hidden = [32, 32]):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets

        self.layer_input = torch.nn.Linear(self.n_features, n_hidden[0])         # Input layer: n_features -> 16
        self.layer_hidden1 = torch.nn.Linear(n_hidden[0], n_hidden[1])           # Hidden layer 1: 16 -> 16
        self.layer_hidden2 = torch.nn.Linear(n_hidden[1], self.n_targets)        # Hidden layer 2: 16 -> n_targets

        self.activation_function = torch.nn.ReLU()

    def forward(self, X):
        output = self.layer_input(X)
        output = self.activation_function(output)
        output = self.layer_hidden1(output)
        output = self.activation_function(output)
        output = self.layer_hidden2(output)
        return output


class NNDropout(torch.nn.Module):
    """Simple feedforward neural network with two hidden layers"""
    def __init__(self, n_features=1, n_targets=1, n_hidden = [32, 32], dropout_rate=0.):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets

        self.layer_input = torch.nn.Linear(self.n_features, n_hidden[0])         # Input layer: n_features -> 16
        self.layer_hidden1 = torch.nn.Linear(n_hidden[0], n_hidden[1])           # Hidden layer 1: 16 -> 16
        self.layer_hidden2 = torch.nn.Linear(n_hidden[1], self.n_targets)        # Hidden layer 2: 16 -> n_targets
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.activation_function = torch.nn.ReLU()

    def forward(self, X):
        output = self.layer_input(X)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_hidden1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_hidden2(output)
        return output