import numpy as np
import torch

class LinearRegression(torch.nn.Module):
    """Linear regression"""
    def __init__(self, n_features=1, n_targets=1, param_dict = None):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_targets)
        if param_dict is not None:
            self.linear.weight = torch.nn.parameter.Parameter(param_dict['linear.weight'])
            self.linear.bias = torch.nn.parameter.Parameter(param_dict['linear.bias'])
            
    def forward(self, X):
        return self.linear(X)

class NN(torch.nn.Module):
    """Neural Network with arbitrary number of layers."""
    def __init__(self, n_features=1, n_targets=1, n_hidden=[16], param_dict = None):
        """Args:
        param_dict (optional) dictionary of weights and biases that must follow size of NN requested
        e.g., to get param_dict from a guide, you can do param_dict = guide.median() or guide()"""
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_hidden = n_hidden
        self.param_dict = param_dict

        nodes = [n_features]+n_hidden+[n_targets]

        self.layers = torch.nn.ModuleList([])
        for j in range(len(nodes)-1):
            linear_j = torch.nn.Linear(nodes[j], nodes[j+1])
            # If guide has been provided, we will set the weights and biases according to the guide values
            if param_dict is not None:
                linear_j.weight = torch.nn.parameter.Parameter(param_dict[f'layers.{j}.weight'])
                linear_j.bias = torch.nn.parameter.Parameter(param_dict[f'layers.{j}.bias'])
            self.layers.append(linear_j)

        self.activation_function = torch.nn.ReLU()

    def forward(self, X):
        for j in range(len(self.layers)-1):
            X = self.layers[j](X)
            X = self.activation_function(X)
        return self.layers[-1](X)


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