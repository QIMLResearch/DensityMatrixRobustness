import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
    
class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.2):
        """
        Initialize a dynamic neural network.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list): List of integers specifying the size of each hidden layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(DynamicNeuralNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Default configuration

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add BatchNorm
            layers.append(nn.ReLU())  # Add activation
            layers.append(nn.Dropout(dropout_rate))  # Add Dropout
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class DynamicDMNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.2):
        """
        Initialize a dynamic neural network.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list): List of integers specifying the size of each hidden layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(DynamicDMNeuralNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Default configuration

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):

            lin = nn.Linear(prev_dim, hidden_dim)
            lin = spectral_norm(lin)   
            layers.append(lin)
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add BatchNorm
            layers.append(nn.ReLU())  # Add activation
            layers.append(nn.Dropout(dropout_rate))  # Add Dropout
            prev_dim = hidden_dim

        # Final output layer
        out = nn.Linear(prev_dim, output_dim)
        out = spectral_norm(out)             
        layers.append(out)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_dynamic_neural_network(
    input_dim, output_dim, multiclass, hidden_dims=None, optimizer="adam", lr=0.001, dropout_rate=0.2
):
    """
    Create a dynamic neural network model.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        multiclass (bool): Whether it's a multi-class problem.
        hidden_dims (list): List of integers specifying the size of each hidden layer.
        optimizer (str): Optimizer to use ("adam" or "sgd").
        lr (float): Learning rate.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        model (nn.Module): The neural network model.
        criterion: The loss function.
        optimizer: The optimizer.
    """

    # Select the criterion based on the task
    if not multiclass:
        output_dim = 2
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model = DynamicNeuralNetwork(input_dim, output_dim, hidden_dims, dropout_rate)

    # Select the optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return model, criterion, optimizer


def create_dynamic_dm_neural_network(
    input_dim, output_dim, multiclass, hidden_dims=None, optimizer="adam", lr=0.001, dropout_rate=0.2
):
    """
    Create a dynamic neural network model.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        multiclass (bool): Whether it's a multi-class problem.
        hidden_dims (list): List of integers specifying the size of each hidden layer.
        optimizer (str): Optimizer to use ("adam" or "sgd").
        lr (float): Learning rate.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        model (nn.Module): The neural network model.
        criterion: The loss function.
        optimizer: The optimizer.
    """

    # Select the criterion based on the task
    if not multiclass:
        output_dim = 2
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model = DynamicDMNeuralNetwork(input_dim, output_dim, hidden_dims, dropout_rate)

    # Select the optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return model, criterion, optimizer