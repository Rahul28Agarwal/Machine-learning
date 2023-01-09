import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_epochs=1000) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None

    def __add_bias(self, X):
        X_b = np.array([[1]]*X.shape[0])
        return np.concatenate((X_b, X), axis=1)

    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y*np.log(h) - (1-y)*np.log(1-h)).mean()

    def gradient_descent_fit(self, X, y):

        y = y.reshape((y.shape[0],1))
        # add bias to input
        X = self.__add_bias(X)
        n_samples, n_features = X.shape

        # randomly initialize weigths
        self.weights = np.random.randn(n_features, 1)

        # gradient descent
        for epoch in range(self.n_epochs):
            # calculate the gradient
            z = np.dot(X, self.weights)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h-y))/y.size
            self.weights -= self.learning_rate * gradient

    def predict_prob(self, X):
        X = self.__add_bias(X)
        return self.__sigmoid(np.dot(X, self.weights))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
