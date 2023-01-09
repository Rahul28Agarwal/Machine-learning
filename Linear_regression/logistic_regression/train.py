from sklearn.datasets  import load_iris
from sklearn.model_selection import train_test_split
import logistic_regression as logit
import numpy as np
iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = logit.LogisticRegression()
log_reg.gradient_descent_fit(X_train, y_train)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # reshape to get a column vector
y_proba = log_reg.predict_prob(X_new)
print(y_proba)



