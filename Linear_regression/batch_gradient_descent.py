import numpy as np

class BatchGradientDescent:
    def __init__(self, learning_rate=0.001, n_epochs=1000) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        #initialize weights and bias with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for epochs in range(self.n_epochs):
            for i in range(n_samples):
                # calculate predictions
                y_pred = np.dot(X[i], self.weights) + self.bias

                # calculate the gradient of the mean squared error loss function with respect to weight and bias
                dw = (1/n_samples) * (y_pred - y[i]) * X[i]
                db = (1/n_samples) * (y_pred - y[i])

                # update the weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def get_weights_bias(self):
        return (self.weights, self.bias)
