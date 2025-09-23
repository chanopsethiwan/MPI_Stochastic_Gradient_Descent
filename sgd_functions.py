import numpy as np
import pandas as pd
import math

def initialise_weights(features, number_of_neurons):
    n_in = features.shape[1]
    
    # Xavier/Glorot uniform initialization
    limit_hidden = np.sqrt(1 / (n_in + number_of_neurons))
    limit_output = np.sqrt(1 / (number_of_neurons + 1))  # +1 for scalar output neuron
    
    weights_dict = {}
    # Hidden layer weights and bias
    weights_dict['weights_features'] = np.random.uniform(-limit_hidden, limit_hidden, (number_of_neurons, n_in))
    weights_dict['bias_features'] = np.zeros(number_of_neurons)  # usually initialised to 0
    weights_dict['weights_after_activation'] = np.random.uniform(-limit_output, limit_output, number_of_neurons)
    weights_dict['bias_after_activation'] = 0.0  # usually initialised to 0
    return weights_dict