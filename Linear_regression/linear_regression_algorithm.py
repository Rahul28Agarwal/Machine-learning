import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_epochs=1000, t0=5, alpha=1.0, l1_ratio=0.5) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.alpha = alpha
        self.t0 = t0
        self.t1 = n_epochs
        self.l1_ratio = l1_ratio
        self.prev_loss = float('inf')

    def learning_schedule(self, t):
        return self.t0/(t + self.t1)
    
    def gradient_descent_fit(self, X, y):
        X_b = np.array([[1]]*X.shape[0])
        X = np.concatenate((X_b, X), axis=1)
        n_samples, n_features = X.shape

        # randomly initialize weights
        self.weights = np.random.randn(n_features,1)


        # gradient descent
        for epoch in range(self.n_epochs):

            # calculate the gradient
            gradient = (2/n_samples) * X.T @(X @ self.weights - y)

            # update the weights and bias
            self.weights -= self.learning_rate * gradient

    
    def stochastic_gradient_descent_fit(self, X, y):
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

    def ridge_GD_fit(self, X, y):
        X_b = np.array([[1]]*X.shape[0])
        X = np.concatenate((X_b, X), axis=1)
        n_samples, n_features = X.shape

        # initialize weights and bias with zeros
        self.weights = np.random.randn(n_features,1)


        # gradient descent
        for epoch in range(self.n_epochs):

            # calculate the gradient
            gradient = (2/n_samples) * X.T @(X @ self.weights - y) + (self.alpha/n_samples)*self.weights

            # update the weights and bias
            self.weights -= self.learning_rate * gradient

    def elastic_net_GD_fit(self, X, y):
        X_b = np.array([[1]]*X.shape[0])
        X = np.concatenate((X_b, X), axis=1)
        n_samples, n_features = X.shape

        # initialize weights and bias with zeros
        self.weights = np.random.randn(n_features,1)


        # gradient descent
        for epoch in range(self.n_epochs):

            # calculate the gradient
            gradient = (2/n_samples) * X.T @(X @ self.weights - y) + (1-self.l1_ratio)*(self.alpha/n_samples)*self.weights + self.l1_ratio*(self.alpha/n_samples)*np.sign(self.weights)

            # update the weights and bias
            self.weights -= self.learning_rate * gradient

    def lasso_GD_fit(self, X, y):
        X_b = np.array([[1]]*X.shape[0])
        X = np.concatenate((X_b, X), axis=1)
        n_samples, n_features = X.shape

        # initialize weights and bias with zeros
        self.weights = np.random.randn(n_features,1)


        # gradient descent
        for epoch in range(self.n_epochs):

            # calculate the gradient
            gradient = (2/n_samples) * X.T @(X @ self.weights - y) + (self.alpha/n_samples)*np.sign(self.weights)

            # update the weights and bias
            self.weights -= self.learning_rate * gradient



    def get_weights_bias(self):
        equation = ''
        for i in range(self.weights.shape[0]):
            if i == 0:
                equation += str(self.weights[i][0])
            if i != 0:
                equation += ' + '
                equation +=  str(self.weights[i][0]) + '*x_' + str(i)
        return equation
