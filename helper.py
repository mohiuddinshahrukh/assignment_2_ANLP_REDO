from model.ffnn import NeuralNetwork, compute_loss
import numpy as np
import matplotlib.pyplot as plt


def batch_train(X, Y, model, train_flag=False, epochs=2000, learning_rate=0.005):
    # Prediction without Training
    predictions_before_training = model.predict(X)
    accuracy_before_training = compute_accuracy(predictions_before_training, Y)
    print(f"Accuracy before training: {accuracy_before_training:.2%}")

    if train_flag:
        # Training
        costs = []
        num_samples = X.shape[1]

        for epoch in range(epochs):
            # Forward pass
            predictions = model.forward(X)
            # Backward pass
            gradients = model.backward(X, Y)
            # Accumulate gradients
            grad_weights_hidden, grad_bias_hidden, grad_weights_output, grad_bias_output = gradients
            # Update weights and biases using accumulated gradients
            model.weights_input_hidden -= learning_rate * grad_weights_hidden / num_samples
            model.bias_hidden -= learning_rate * grad_bias_hidden / num_samples
            model.weights_hidden_output -= learning_rate * grad_weights_output / num_samples
            model.bias_output -= learning_rate * grad_bias_output / num_samples

            # Compute average loss for the epoch
            average_loss = compute_loss(predictions, Y)
            costs.append(average_loss)

        # Prediction after Training
        predictions_after_training = model.predict(X)
        accuracy_after_training = compute_accuracy(predictions_after_training, Y)
        print(f"Accuracy after training: {accuracy_after_training:.2%}")

        # Plot the Cost Function
        plot_cost_function(costs, title='Cost Function During Training')


def minibatch_train(X, Y, model, train_flag=True, epochs=1000, learning_rate=0.0001, batch_size=64):
    if train_flag:
        # Training
        costs = []
        num_samples = X.shape[1]

        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]

                # Forward pass
                predictions = model.forward(X_batch)
                # Backward pass
                gradients = model.backward(X_batch, Y_batch)
                # Accumulate gradients
                grad_weights_hidden, grad_bias_hidden, grad_weights_output, grad_bias_output = gradients
                # Update weights and biases using accumulated gradients
                model.weights_input_hidden -= learning_rate * grad_weights_hidden / batch_size
                model.bias_hidden -= learning_rate * grad_bias_hidden / batch_size
                model.weights_hidden_output -= learning_rate * grad_weights_output / batch_size
                model.bias_output -= learning_rate * grad_bias_output / batch_size

            # Compute average loss for the epoch
            predictions_after_training = model.predict(X)
            average_loss = compute_loss(predictions_after_training, Y)
            costs.append(average_loss)

        # Plot the Cost Function
        plot_cost_function(costs, title='Cost Function During Mini-batch Training')


def compute_accuracy(predictions, ground_truth):
    # Your accuracy calculation logic goes here
    # Example: Calculate accuracy as the percentage of correct predictions
    return np.mean(np.argmax(predictions, axis=0) == np.argmax(ground_truth, axis=0))


def plot_cost_function(costs, title='Cost Function During Training'):
    plt.plot(costs)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()