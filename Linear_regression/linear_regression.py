# implemeting the linear regression using gradient descent.
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_epochs=1000) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None

    def fit(self, X, y):
        X_b = np.array([[1]]*X.shape[0])
        X = np.concatenate((X_b, X), axis=1)
        n_samples, n_features = X.shape

        # initialize weights and bias with zeros
        self.weights = np.random.randn(n_features,1)


        # gradient descent
        for epoch in range(self.n_epochs):

            # calculate the gradient
            gradient = (2/n_samples) * X.T @(X @ self.weights - y)

            # update the weights and bias
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def get_weights_bias(self):
        equation = ''
        for i in range(self.weights.shape[0]):
            if i == 0:
                equation += str(self.weights[i][0])
            if i != 0:
                equation += ' + '
                equation +=  str(self.weights[i][0]) + '*x_' + str(i)
        return equation

