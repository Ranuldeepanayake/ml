#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Data set column legend
columnSubscribers = 2; # subscribers[2]
columnVideoViews = 3;  # video views[3]

#Reading the dataset from the file.
dataset = pd.read_csv("./encoded-spotify-2023.csv")

## Data preprocessing
dataset.head()

# Check the structure of the data set (rows and columns).
dataset.shape

# Check for missing values.
dataset.isna().sum()

# Check for duplicate values.
dataset.duplicated().any()

## End of data preprocessing

# Create plots.
fig, axs = plt.subplots(1, figsize = (5,5))
plt.tight_layout()

# Create the subscriber distribution graph.
sns.distplot(dataset['in_spotify_playlists']);

# Show the relationships between sales and modes of advertising.
sns.pairplot(dataset, x_vars=['in_spotify_playlists'], y_vars='in_apple_playlists', height=4, aspect=1, kind='scatter')
plt.show()

# Model building.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

## Simple linear regression (model #1).

#Setting the value for X and Y
x = dataset[['in_spotify_playlists']]
y = dataset['in_apple_playlists']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

slr= LinearRegression()  
slr.fit(x_train, y_train)

#Printing the model coefficients
print('Intercept: ', slr.intercept_)
print('Coefficient:', slr.coef_)
print('Regression Equation: Views = ', slr.intercept_, ' + ', slr.coef_, ' * Subscribers')

#Show the line of best fit
plt.scatter(x_train, y_train)
plt.plot(x_train, 6.948 + 0.054*x_train, 'r')
plt.show()

#Prediction of Test and Training set result  
y_pred_slr= slr.predict(x_test)  
x_pred_slr= slr.predict(x_train)  
print("Prediction for test set: {}".format(y_pred_slr))

#Actual value and the predicted value
slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
slr_diff

#Predict for any value
slr.predict([[56]])

# print the R-squared value for the model
from sklearn.metrics import accuracy_score
print('R squared value of the model: {:.2f}'.format(slr.score(x,y)*100))

# 0 means the model is perfect. Therefore the value should be as close to 0 as possible
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

"""

# Multiple linear regression (model #2).

#Setting the value for X and Y
x = dataset[['TV', 'Radio', 'Newspaper']]
y = dataset['Sales']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=100)  

mlr= LinearRegression()  
mlr.fit(x_train, y_train) 

#Printing the model coefficients
print(mlr.intercept_)
# pair the feature names with the coefficients
list(zip(x, mlr.coef_))

#Predicting the Test and Train set result 
y_pred_mlr= mlr.predict(x_test)  
x_pred_mlr= mlr.predict(x_train) 
print("Prediction for test set: {}".format(y_pred_mlr))

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff

#Predict for any value
mlr.predict([[56, 55, 67]])

# print the R-squared value for the model
print('R squared value of the model: {:.2f}'.format(mlr.score(x,y)*100))

# 0 means the model is perfect. Therefore the value should be as close to 0 as possible
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr) """