import numpy as np
import sys


class Perceptron:
    def __init__(self, num_features, learning_rate=0.01):
        # Initialize weights to a constant value (e.g., zero)
        self.weights = np.zeros(num_features)
        # Initialize threshold to a constant value (e.g., zero)
        self.threshold = 0.01
        self.learning_rate = learning_rate


    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.threshold
        return 1 if activation >= 0 else 0

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        target = int(target)  # Convert target to integer
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        self.threshold += self.learning_rate * error


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', dtype=str)
    X = data[:, :-1].astype(float)  # Features
    y_str = data[:, -1]  # Labels

    # Map string labels to integer values
    label_map = {label: i for i, label in enumerate(np.unique(y_str))}
    y = np.array([label_map[label] for label in y_str])

    return X, y



def calculate_accuracy(perceptron, test_X, test_y):
    correct = 0
    total = len(test_y)
    for i in range(total):
        prediction = perceptron.predict(test_X[i])
        if prediction == test_y[i]:
            correct += 1
    accuracy = (correct / total) * 100
    return accuracy


def main(train_file, test_file, learning_rate, epochs):
    train_X, train_y = load_data(train_file)
    test_X, test_y = load_data(test_file)

    num_features = train_X.shape[1]
    perceptron = Perceptron(num_features, learning_rate)

    for epoch in range(epochs):
        # Shuffle training data
        shuffled_indices = np.random.permutation(len(train_y))
        train_X = train_X[shuffled_indices]
        train_y = train_y[shuffled_indices]

        # Train perceptron
        for i in range(len(train_y)):
            perceptron.train(train_X[i], train_y[i])

        # Calculate and print accuracy on test set
        accuracy = calculate_accuracy(perceptron, test_X, test_y)
        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%')

    # Prediction loop
    while True:
        new_observation = input("Enter new observations (comma-separated values), or type 'exit' to quit: ")
        if new_observation.lower() == 'exit':
            break
        new_observation = np.array([float(val) for val in new_observation.split(',')])
        prediction = perceptron.predict(new_observation)
        print(f'Predicted class: {prediction}')


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python perceptron.py")
        sys.exit(1)

    train_file = input("Enter path to file with training data: ")
    test_file = input("Enter path to file with test data: ")
    learning_rate = float(input("Enter learning rate: "))
    epochs = int(input("Enter number of epochs: "))

    main(train_file, test_file, learning_rate, epochs)
