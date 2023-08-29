# https://realpython.com/linear-regression-in-python/

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)

print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
