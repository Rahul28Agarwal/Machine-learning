import numpy as np

class StochasticGradientDescent:
    def __init__(self, learning_rate=0.001, n_epochs=50, t0=5) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.t0 = t0
        self.t1 = n_epochs

    def learning_schedule(self, t):
        return self.t0/(t + self.t1)
    
    def fit(self, X, y):
        ones = np.ones(shape=(X.shape[0],1))
        X = np.concatenate((ones, X), axis=1)
        n_samples, n_features = X.shape

        # initialize weights
        self.weights = np.random.randn(n_features,1)

        # gradient descent
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # random index
                random_index = np.random.randint(n_samples)

                # X and y value for the random index
                Xi = X[random_index: random_index + 1]
                yi = y[random_index: random_index+1]

                # gradient 
                gradients = 2 * Xi.T @ (Xi @ self.weights - yi)

                self.learning_rate = self.learning_schedule(epoch * n_samples + i)

                self.weights -= self.learning_rate * gradients

    def predict(self, X):
        ones = np.ones(shape=(X.shape[0],1))
        X = np.concatenate((ones, X), axis=1)
        y_pred = np.dot(X, self.weights)
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
