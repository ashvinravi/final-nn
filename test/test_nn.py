# TODO: import dependencies and write unit tests below
import numpy as np
from nn import NeuralNetwork
from numpy.typing import ArrayLike
from nn import io, preprocess
import pytest

# REWRITING RELU AND sigmoid RELU_BACKPROP functions just for using for test cases. 
def relu(Z: ArrayLike) -> ArrayLike:
    return np.maximum(0, Z)

def sigmoid(Z: ArrayLike) -> ArrayLike:
    nl_transform = 1 / (1 + np.exp(-Z))
    return nl_transform

def sigmoid_backprop(dA: ArrayLike, Z: ArrayLike):
    dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))
    return dZ

def test_single_forward():
    
    # test single forward pass for neural network with input layer (3) and output layer (4). 
    c = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 4, 'activation': '_relu'}],
        lr=0.01,
        seed=150,
        batch_size=1,
        epochs=1,
        loss_function="_mean_squared_error")

    input_matrix = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]])

    weights = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]])

    biases = np.array([0.1, 0.5, 0.2, 0.3])

    Z = np.dot(input_matrix, weights.T) + biases
    correct_single_forward_pass = relu(Z)
    calculated_single_forward_pass = c._single_forward(weights.T, biases, input_matrix, '_relu')
    assert ( np.allclose(calculated_single_forward_pass, correct_single_forward_pass ))

def test_forward():
    # Testing forward pass for a neural network with input layer (3), hidden layer (5), and output (4). 
    c = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 5, 'activation': '_relu'},
                               {'input_dim': 5, 'output_dim': 4, 'activation': '_relu'}],
        lr=0.01,
        seed=150,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error")
    
    input_matrix_input_hidden = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.6, 0.4, 0.7],
        [0.5, 0.1, 0.2]])
    input_hidden_weights = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.2, 0.1, 0.4],
        [0.1, 0.3, 0.4],
        [0.6, 0.8, 0.4]])
    input_hidden_biases = np.array([0.1, 0.5, 0.2, 0.3, 0.2])

    hidden_output_weights = np.array([[0.5, 0.2, 0.3, 0.4, 0.1], 
                                      [0.1, 0.2, 0.4, 0.3, 0.6], 
                                      [0.7, 0.2, 0.9, 0.4, 0.1], 
                                      [0.05, 0.95, 0.4, 0.7, 0.8]])
    hidden_output_biases = np.array([0.8, 0.65, 0.6, 0.4])
    
    c._param_dict['W1'] = input_hidden_weights 
    c._param_dict['b1'] = input_hidden_biases
    c._param_dict['W2'] = hidden_output_weights
    c._param_dict['b2'] = hidden_output_biases
    
    matrix_hidden_output = relu(np.dot(input_matrix_input_hidden, input_hidden_weights.T) + input_hidden_biases)
    correct_forward_pass = relu(np.dot(matrix_hidden_output, hidden_output_weights.T) + hidden_output_biases)
    calculated_forward_pass, _ = c.forward(input_matrix_input_hidden)
    assert ( np.allclose(correct_forward_pass, calculated_forward_pass) )
      
def test_single_backprop():
    # test single backprop for neural network with input layer (3) and output layer (4). 
    c = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 4, 'activation': '_relu'}],
        lr=0.01,
        seed=150,
        batch_size=1,
        epochs=1,
        loss_function="mean_squared_error")
    
    # Z_curr is output matrix 
    output_matrix = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]])

    weights = np.array([[0.1, 0.2, 0.3, 0.4],
        [0.4, 0.5, 0.6, 0.3],
        [0.7, 0.8, 0.9, 0.2]])
    
    A_prev = np.array([[0.7, 0.9, 0.2],
        [0.6, 0.3, 0.7],
        [0.95, 0.01, 0.72],
        [0.66, 0.44, 0.12]])
    
    dA_curr = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]])

    biases = np.array([0.1, 0.5, 0.2])

    derivative_activation_curr = sigmoid_backprop(dA_curr, output_matrix)        
        
    correct_dW_curr = (np.dot(A_prev.T, derivative_activation_curr) / A_prev.shape[1]).T
    correct_db_curr = np.sum(derivative_activation_curr, axis=1, keepdims=True) / A_prev.shape[1]
    correct_dA_prev = np.dot(derivative_activation_curr, weights) 

    calculated_dW, calculated_db, calculated_dA = c._single_backprop(weights, biases, output_matrix, A_prev, dA_curr, '_sigmoid')
   
    assert ( np.allclose( calculated_dW, correct_dW_curr ))
    assert ( np.allclose( calculated_db, correct_db_curr ))
    assert ( np.allclose( calculated_dA, correct_dA_prev ))


def test_predict():
        # test single forward pass for neural network with input layer (3) and output layer (4). 
    c = NeuralNetwork(nn_arch=[{'input_dim': 3, 'output_dim': 4, 'activation': '_relu'}],
        lr=0.01,
        seed=150,
        batch_size=1,
        epochs=1,
        loss_function="_mean_squared_error")

    input_matrix = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]])

    weights = np.array([[0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4]])

    biases = np.array([0.1, 0.5, 0.2, 0.3])

    c._param_dict['W1'] = weights
    c._param_dict['b1'] = biases

    calculated_prediction = c.predict(input_matrix)
    Z = np.dot(input_matrix, weights.T) + biases
    correct_prediction = relu(Z)

    assert ( np.allclose(calculated_prediction, correct_prediction ))
    


    # Z = np.dot(input_matrix, weights.T) + biases
    #correct_single_forward_pass = relu(Z)
    #calculated_single_forward_pass = c._single_forward(weights.T, biases, input_matrix, '_relu')
    #assert ( np.allclose(calculated_single_forward_pass, correct_single_forward_pass ))

    


def test_binary_cross_entropy():
    y_hat = np.array([0.8, 0.3, 0.95, 0.9])
    y = np.array([1, 0, 1, 1])

    a = NeuralNetwork(nn_arch=[{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, 
                           {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                  lr=0.01,
                  seed=150,
                  batch_size=1,
                  epochs=1,
                  loss_function="mean_squared_error")

    calculated_loss = a._binary_cross_entropy(y, y_hat)
    correct_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    assert ( round(correct_loss, 3) == round(calculated_loss, 3) )

def test_binary_cross_entropy_backprop():
    y_hat = np.array([0.8, 0.3, 0.95, 0.9])
    y = np.array([1, 0, 1, 1])

    a = NeuralNetwork(nn_arch=[{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, 
                           {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                  lr=0.01,
                  seed=150,
                  batch_size=1,
                  epochs=1,
                  loss_function="mean_squared_error")
    

    calculated_loss_backprop = a._binary_cross_entropy_backprop(y, y_hat)
    correct_loss_backprop = (y_hat - y) / ((1 - y_hat) * y_hat)

    assert ( np.allclose(calculated_loss_backprop, correct_loss_backprop) )

def test_mean_squared_error():

    a = NeuralNetwork(nn_arch=[{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, 
                           {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                  lr=0.01,
                  seed=150,
                  batch_size=1,
                  epochs=1,
                  loss_function="mean_squared_error")
    
    y_hat = np.array([16, 18, 14, 19])
    y = np.array([17.2, 16.8, 14.3, 18.6]) 
    calculated_MSE = a._mean_squared_error(y_hat, y)
    correct_MSE = np.mean(np.square(y_hat - y))

    assert ( round(calculated_MSE, 3) == round(correct_MSE, 3) )

def test_mean_squared_error_backprop():
    a = NeuralNetwork(nn_arch=[{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, 
                           {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                  lr=0.01,
                  seed=150,
                  batch_size=1,
                  epochs=1,
                  loss_function="mean_squared_error")
    
    y_hat = np.array([16, 18, 14, 19])
    y = np.array([17.2, 16.8, 14.3, 18.6]) 
    calculated_MSE = a._mean_squared_error(y_hat, y)
    correct_MSE = np.mean(np.square(y_hat - y))

    assert ( np.allclose(calculated_MSE, correct_MSE) )

def test_sample_seqs():
    # Given two very imbalanced classes, return correct sampling scheme. 
    # NOTE: the function assumes that there will be many more negatives than positives, and positives will be much smaller - so it checks for one-sided imbalance
    sequences = ['AGAG', 'CCAG', 'ATAG', 'GGTA', "GCCA", "ATTA", "TATA", "TCAT", "TCAA", "AAGC", "TTTT"]
    labels = [True, False, False, False, True, False, False, False, True, False, False]

    sampled_sequences, sampled_labels = preprocess.sample_seqs(sequences, labels)

    # sample_seqs should return 3 positive sequences and 3 randomly sampled negative sequences. 
    assert ( (len(sampled_labels) - sum(sampled_labels)) == 3 )

def test_one_hot_encode_seqs():
    seq = "ATAG"
    correct_one_hot_encode = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    assert ( np.allclose( preprocess.one_hot_encode_seqs(seq), correct_one_hot_encode ) )

