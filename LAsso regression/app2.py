import numpy as np

class LassoRegression:
    def __init__(self, learning_rate, lambda_param, num_iterations):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_pred = self.predict(X)

            # Calculate gradients
            d_weights = (1 / n_samples) * (X.T.dot(y_pred - y)) + (self.lambda_param / n_samples) * np.sign(self.weights)
            d_bias = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

    def predict(self, X):
        return X.dot(self.weights) + self.bias
