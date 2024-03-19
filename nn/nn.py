# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import random

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]], # type: ignore
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # no transposing of weights here so weights are transposed in the forward function instead. 
        activation_funct = getattr(NeuralNetwork, activation)

        # current_z = current_weight * previous_A + current_basis
        # current_a = activation(current_z)
        Z_curr = np.dot(A_prev, W_curr) + b_curr
        A_curr = activation_funct(self, Z_curr)

        return A_curr, Z_curr
        

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        A_curr = X

        for layer, dict in enumerate(self.arch):
            # add one since python indexing is at zero but weights initialized at W1
            weight_index = str(layer + 1)
            # print("Layer" + weight_index + "Propogation")

            # transpose weights for dot product multiplication 
            current_weights = self._param_dict['W' + weight_index].T
            # same thing with bias, transpose weights 
            current_bias = self._param_dict['b' + weight_index].T
            # pull activation function 
            activation_function = dict['activation']

            # Store cache before forward pass -> import for backpropogation. 
            cache['A' + weight_index] = A_curr

            # run forward pass on layer 
            A_curr, Z_curr = self._single_forward(current_weights, current_bias, A_curr, activation_function)

            # Store original Z 
            cache['Z' + weight_index] = Z_curr

        return A_curr, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias zmatrix.
        """
        # Calculate backprop of dA_curr depending on what activation function of current layer is
        if activation_curr == '_sigmoid':
            dA_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            dA_curr = self._relu_backprop(dA_curr, Z_curr)

        dW_curr = (np.dot(A_prev.T, dA_curr) / A_prev.shape[1]).T
        db_curr = np.sum(dA_curr, axis=1, keepdims=True) / A_prev.shape[1]
        dA_prev = np.dot(dA_curr, W_curr)
        
        return dW_curr, db_curr, dA_prev

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """

        grad_dict = {}

        if self._loss_func == '_mean_squared_error':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)


        for i in reversed(range(0, len(self.arch))):
            # weights/dictionary is indexed from 1 not 0. 
            layer = i + 1

            A_prev = cache['A' + str(layer)]
            Z_curr = cache['Z' + str(layer)]

            # pull activation function 
            dictionary = self.arch[i]
            activation = dictionary['activation']

            # pull weights for each layer 
            W_curr = self._param_dict['W' + str(layer)]
            b_curr = self._param_dict['b' + str(layer)]

            # run backpropagation for current layer
            dW_curr, db_curr, dA_prev = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)
            
            # update gradient dictionary
            grad_dict['dW' + str(layer)] = dW_curr
            grad_dict['db' + str(layer)] = db_curr

            # update dA_curr before moving onto next layer 
            dA_curr = dA_prev

        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for key in grad_dict:
            if key.startswith('dW'):
                param_name = 'W' + key[2:]
                # subtract gradients for current layer from weights to get most updated weights 
                self._param_dict[param_name] -= (self._lr * grad_dict[key])
            else:
                param_name = 'b' + key[2:]
                # Calculate the mean gradient for bias across each batch
                mean_grad = np.mean(grad_dict[key], axis=0, keepdims=True)
                # subtract mean gradient for current layer from bias to get most updated biases for current layer 
                self._param_dict[param_name] -= (self._lr * mean_grad)
            

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        per_epoch_loss_train = []
        per_epoch_loss_validation = []

        # loop through each epoch 
        for epoch in range(self._epochs):
            epoch_losses = []
            # split training data into mini batches randomly after shuffling
            array = list(np.arange(len(X_train)))
            indices = np.array(random.sample(array, len(array)))

            # loop through each mini-batch 
            for i in range(0, len(X_train), self._batch_size):
                # forward pass
                batch_indices = indices[i:i+self._batch_size]
                x_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Forward pass 
                pred, hidden_layers = self.forward(x_batch)

                # calculate loss for forward pass 
                # y_hat = pred, y_actual = y_batch
                if self._loss_func == "_binary_cross_entropy":
                    loss = self._binary_cross_entropy(y_batch, pred)
                else:
                    loss = self._mean_squared_error(y_batch, pred)

                epoch_losses.append(loss)

                # backpropogation here 
                grad_dict = self.backprop(y_batch, pred, hidden_layers)
                
                # update weights and biases after backpropogation 
                self._update_params(grad_dict)

            # Calculate validation loss for this epoch
            valid_preds, _ = self.forward(X_val)
            if self._loss_func == "_binary_cross_entropy":
                valid_loss = self._binary_cross_entropy(valid_preds, y_val)
            else:
                valid_loss = np.mean(self._mean_squared_error(valid_preds, y_val))
            
            # Append training and validation losses to respective lists
            train_loss = np.mean(epoch_losses)
            per_epoch_loss_train.append(train_loss)
            per_epoch_loss_validation.append(valid_loss)        
        return per_epoch_loss_train, per_epoch_loss_validation 

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # cache is not needed here, you use prediction for predict()
        prediction, _ = self.forward(X)
        return prediction

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # From HW 7: returns 1/(1 + e^(-z))
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # Derivative of sigmoid is y = sigmoid(x) * (1 - sigmoid(x))
        # Multiply dA by derivative of sigmoid to get dZ. 
        dZ = dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dZ 


    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0, Z)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # return derivative of _relu function 
        # f(x) = 0 when x <= 0, so f'(x) = 0, f(x) = x when x > 0 so f'(x) = 1
        derivative = np.where(Z <= 0, 0, 1)
        
        dZ = dA * derivative
        return dZ 
        

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Taken from HW 7 function 
        # If y = 0, then your left side equals 0. 
        y0 = y * np.log(y_hat)
        # If y = 1, then your right side equals 0. 
        y1 = (1 - y) * np.log(1 - y_hat)

        loss_vector = - (y0 + y1)
        loss = np.mean(loss_vector)

        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = (y_hat - y) / ((1 - y_hat) * y_hat)
        return dA 

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean(np.square(y - y_hat))

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = -2 * (y - y_hat)
        return dA
