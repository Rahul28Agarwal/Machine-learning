import numpy as np
import matplotlib.pyplot as plt
import linear_regression as lr
import stochastic as slr

np.random.seed(42)
n_samples = 100
X = 2* np.random.rand(n_samples,1)
y = 4 + 3* X + np.random.randn(n_samples, 1)

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])
ax.grid(alpha=0.5)
ax.scatter(X, y, color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Randomly generated linear dataset')
ax.set_ylim(ymin=0)

plt.show()


model = lr.LinearRegression()
model.fit(X,y)
print("model works")
print(model.get_weights_bias())
# model.predict([])

slr_model = slr.StochasticGradientDescent()
slr_model.fit(X,y)
print(slr_model.get_weights_bias())