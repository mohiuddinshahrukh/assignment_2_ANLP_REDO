import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            seed: int = 1
    ):
        np.random.seed(seed)
        self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-1, 1, (hidden_size, 1))
        self.weights_hidden_output = np.random.uniform(-1, 1, (num_classes, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (num_classes, 1))
        self.num_classes = num_classes

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        z_hidden = np.dot(self.weights_input_hidden, X) + self.bias_hidden
        a_hidden = relu(z_hidden)

        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        a_output = softmax(z_output)

        return a_output

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        predictions = self.forward(X)
        indices = np.argmax(predictions, axis=0)
        predicted_labels = [np.eye(self.num_classes)[index] for index in indices]
        return np.array(predicted_labels).T

    def backward(
            self,
            X: npt.ArrayLike,
            Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        z_hidden = np.dot(self.weights_input_hidden, X) + self.bias_hidden
        a_hidden = relu(z_hidden)
        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        a_output = softmax(z_output)

        # Compute loss
        loss = compute_loss(a_output, Y)
        # print("LOSS: ", loss)

        # Backward pass
        delta_output = a_output - Y
        delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * relu_prime(z_hidden)

        # Compute gradients
        grad_weights_output = np.dot(delta_output, a_hidden.T)
        grad_bias_output = np.sum(delta_output, axis=1, keepdims=True)
        grad_weights_hidden = np.dot(delta_hidden, X.T)
        grad_bias_hidden = np.sum(delta_hidden, axis=1, keepdims=True)

        return grad_weights_hidden, grad_bias_hidden, grad_weights_output, grad_bias_output


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    num_samples = pred.shape[1]

    # Clip values to avoid log(0)
    epsilon = 1e-15
    predictions = np.clip(pred, epsilon, 1 - epsilon)

    # Transpose predictions for correct shape
    predictions = predictions.T

    # Cross-entropy loss calculation
    loss = -np.sum(truth * np.log(predictions.T)) / num_samples

    return loss
