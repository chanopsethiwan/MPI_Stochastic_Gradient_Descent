import numpy as np
import pandas as pd
import math


def random_batch(X, y, batch_size=100):
    random_indices = np.random.choice(X.shape[0], batch_size, replace=False)
    return list(zip(X[random_indices], y[random_indices]))


# optional, you can make your own customised one
def create_batches(X, y, batch_size=100):
    """
    Split dataset into batches of (X, y) pairs.

    Args:
        X (np.ndarray or list): Features.
        y (np.ndarray or list): Labels.
        batch_size (int): Number of samples per batch.

    Returns:
        list: A list of batches, where each batch is a list of (x, y) tuples.
    """
    n_samples = len(X)
    n_batches = math.ceil(n_samples / batch_size)

    data_batch_list = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = list(zip(X[start:end], y[start:end]))
        data_batch_list.append(batch)
    return data_batch_list


# This function initializes the weights for a simple feedforward neural network with one hidden layer.
# The weights are initialized using Xavier/Glorot uniform initialization.
def initialise_weights(features, number_of_neurons):
    n_in = features.shape[1]

    # Xavier/Glorot uniform initialization
    limit_hidden = np.sqrt(6 / (n_in + number_of_neurons))
    limit_output = np.sqrt(6 / (number_of_neurons + 1))  # +1 for scalar output neuron

    weights_dict = {}
    # Hidden layer weights and bias
    weights_dict["weights_features"] = np.random.uniform(
        -limit_hidden, limit_hidden, (number_of_neurons, n_in)
    )
    weights_dict["bias_features"] = np.zeros(
        number_of_neurons
    )  # usually initialised to 0
    weights_dict["weights_after_activation"] = np.random.uniform(
        -limit_output, limit_output, number_of_neurons
    )
    weights_dict["bias_after_activation"] = 0.0  # usually initialised to 0
    return weights_dict


# Define the activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def diff_relu(x):
    return np.where(x > 0, 1, 0)


def diff_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def diff_tanh(x):
    return 1 - np.tanh(x) ** 2


# Now define the function to calculate gradients and loss (combined for efficiency)


def calculate_diff_and_gradients(weights_dict, data_batch, activation_func):
    """
    Computes the difference (error) for each sample and the average gradients
    with respect to model parameters for the given activation function.

    Parameters:
        weights_dict (dict): Dictionary containing model parameters:
        'weights_features', 'bias_features', 'weights_after_activation', 'bias_after_activation'.
        data_batch (list): List of tuples (features_list, actual_val).
        activation_func (str): Activation function to use: 'relu', 'sigmoid', or 'tanh'.

    Returns:
        diffs (list): Errors (final_output - actual_val) for each sample in the batch.
        gradient_dict (dict): Averaged gradients for each parameter.
    """

    # Initialize gradient accumulators with same shapes as weights/biases
    grad_weights_features = np.zeros_like(
        weights_dict["weights_features"], dtype="float64"
    )
    grad_bias_features = np.zeros_like(weights_dict["bias_features"], dtype="float64")
    grad_weights_after_activation = np.zeros_like(
        weights_dict["weights_after_activation"], dtype="float64"
    )
    grad_bias_after_activation = 0.0  # bias_after_activation is a scalar

    diffs = []
    batch_size = len(data_batch)

    for features_list, actual_val in data_batch:
        # Forward pass: compute linear combination z and activation
        z = (
            np.dot(weights_dict["weights_features"], features_list)
            + weights_dict["bias_features"]
        )

        if activation_func == "relu":
            activated_output = relu(z)
            activated_derivative = diff_relu(z)
        elif activation_func == "sigmoid":
            activated_output = sigmoid(z)
            activated_derivative = diff_sigmoid(z)
        elif activation_func == "tanh":
            activated_output = tanh(z)
            activated_derivative = diff_tanh(z)
        else:
            raise ValueError("Unsupported activation function")

        # Compute final output and error (difference from actual value)
        final_output = (
            np.dot(weights_dict["weights_after_activation"], activated_output)
            + weights_dict["bias_after_activation"]
        )
        error = final_output - actual_val
        diffs.append(error)

        # Gradient w.r.t output layer weights and bias
        grad_weights_after_activation += error * activated_output
        grad_bias_after_activation += error

        # Gradient w.r.t hidden layer weights and bias
        delta_hidden = (
            error * weights_dict["weights_after_activation"] * activated_derivative
        )
        grad_weights_features += np.outer(delta_hidden, features_list)
        grad_bias_features += delta_hidden

    # Average gradients over the batch
    gradient_dict = {
        "grad_weights_features": grad_weights_features / batch_size,
        "grad_bias_features": grad_bias_features / batch_size,
        "grad_weights_after_activation": grad_weights_after_activation / batch_size,
        "grad_bias_after_activation": grad_bias_after_activation / batch_size,
    }
    loss = sum((x**2 for x in diffs)) / batch_size
    return loss, gradient_dict


# Update the weights using SGD, gradients come from calculate_gradients function
def update_weights(weights_dict, gradients, learning_rate):
    # Update weights and biases using the computed gradients
    weights_dict["weights_features"] -= (
        learning_rate * gradients["grad_weights_features"]
    )
    weights_dict["bias_features"] -= learning_rate * gradients["grad_bias_features"]
    weights_dict["weights_after_activation"] -= (
        learning_rate * gradients["grad_weights_after_activation"]
    )
    weights_dict["bias_after_activation"] -= (
        learning_rate * gradients["grad_bias_after_activation"]
    )
    return weights_dict


def calculate_loss(weights_dict, data_batch, activation_func):
    loss = 0
    for features_list, actual_val in data_batch:
        err_sq = (
            evaluate(weights_dict, features_list, activation_func) - actual_val
        ) ** 2
        loss += err_sq
    return loss


# , len(
#         data_batch
#     )  # returns the total loss and number of samples in the mini batch in this process


def evaluate(weights_dict, features_list, activation_func):
    # perform matrix multiplication of features and weights
    z = (
        np.dot(weights_dict["weights_features"], features_list)
        + weights_dict["bias_features"]
    )
    # apply activation function
    if activation_func == "relu":
        activated_output = relu(z)
    elif activation_func == "sigmoid":
        activated_output = sigmoid(z)
    elif activation_func == "tanh":
        activated_output = tanh(z)
    else:
        raise ValueError("Unsupported activation function")

    # calculate final output
    return (
        np.dot(weights_dict["weights_after_activation"], activated_output)
        + weights_dict["bias_after_activation"]
    )
