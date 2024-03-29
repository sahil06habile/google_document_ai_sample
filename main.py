import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset (replace with your own dataset)
# Assume X represents the independent variable and y represents the dependent variable
# X = np.array([[1], [2], [3], [4], [5]])  # First variable
X = np.array([0.80, 0.85, -0.79, 0.98, 8])  # First variable
y = np.array([2.30, 2.40, 6.16, 2.78, 3.0])  # Second variable

X = X.reshape(-1, 1)


# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the second variable for a new value of the first variable
new_X = np.array([[0.69]])  # New value of the first variable
predicted_y = model.predict(new_X)

print("Predicted value of the second variable:", predicted_y[0])
