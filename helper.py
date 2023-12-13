from model.ffnn import NeuralNetwork, compute_loss
import numpy as np
import matplotlib.pyplot as plt


def batch_train(X, Y, model, train_flag=False, epochs=1000, learning_rate=0.005):
    # Prediction without Training
    predictions_before_training = model.predict(X)
    accuracy_before_training = compute_accuracy(predictions_before_training, Y)

    print(f"Accuracy before training: {accuracy_before_training}")

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

            pred_to_acc = model.predict(X)
            acc = compute_accuracy(pred_to_acc, Y)
            print(f"Accuracy in epoch {epoch}: {acc}, loss: {average_loss}")

        # Prediction after Training
        predictions_after_training = model.predict(X)
        accuracy_after_training = compute_accuracy(predictions_after_training, Y)
        print(f"Accuracy after training: {accuracy_after_training}")

        # Plot the Cost Function
        plot_cost_function(costs, title='Cost Function During Training')


def minibatch_train(X, Y, model, train_flag=True, epochs=1000, learning_rate=0.005, batch_size=64):
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
            accuracy_after_training = compute_accuracy(predictions_after_training, Y)
            average_loss = compute_loss(predictions_after_training, Y)
            costs.append(average_loss)
            print(f"Accuracy in epoch {epoch}: {accuracy_after_training}, loss: {average_loss}")


        print(f"Accuracy after training: {accuracy_after_training}")

        # Plot the Cost Function
        plot_cost_function(costs, title='Cost Function During Mini-batch Training')


def compute_accuracy(predictions, ground_truth):
    # Your accuracy calculation logic goes here
    # Example: Calculate accuracy as the percentage of correct prediction
    # there was a problem with the code: that predictions parameter Was a number,
    # it should be a vector here instead to correctly compare
    # it with ground_truth which is a matrix
    correct = [1 for index in range(ground_truth.shape[1])
               if np.all(predictions[:, index] == ground_truth[:, index], axis=0)]

    number = sum(correct) / ground_truth.shape[1]
    return number


def plot_cost_function(costs, title='Cost Function During Training'):
    plt.plot(costs)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()
