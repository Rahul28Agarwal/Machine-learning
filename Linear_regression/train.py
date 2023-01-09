import numpy as np
import matplotlib.pyplot as plt
import linear_regression_algorithm as lr

np.random.seed(42)
n_samples = 100
X = 2* np.random.rand(n_samples,1)
y = 4 + 3* X + np.random.randn(n_samples, 1)

# fig=plt.figure()

# ax=fig.add_axes([0,0,1,1])
# ax.grid(alpha=0.5)
# ax.scatter(X, y, color='b')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Randomly generated linear dataset')
# ax.set_ylim(ymin=0)

# plt.show()

model1 = lr.LinearRegression(n_epochs=5000)
model1.gradient_descent_fit(X,y)
print(model1.get_weights_bias())
model2 = lr.LinearRegression(n_epochs=50)
model2.stochastic_gradient_descent_fit(X,y)
print(model2.get_weights_bias())
model1.ridge_GD_fit(X,y)
print(model1.get_weights_bias())
model1.lasso_GD_fit(X,y)
print(model1.get_weights_bias())
model1.elastic_net_GD_fit(X,y)
print(model1.get_weights_bias())