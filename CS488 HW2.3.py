import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

# load iris data
iris = load_iris()

# Creating pd DataFrames
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target_df = pd.DataFrame(data=iris.target, columns=['species'])


# generate labels
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'


target_df['species'] = target_df['species'].apply(converter)

# concatenate the dataframes
iris_df = pd.concat([iris_df, target_df], axis=1)

# iris data statistics
print(iris_df.describe())

from sklearn.metrics import mean_squared_error, r2_score

# converting objects to numerical datatype
iris_df.drop('species', axis=1, inplace=True)
target_df = pd.DataFrame(columns=['species'], data=iris.target)
iris_df = pd.concat([iris_df, target_df], axis=1)

# Variables
X = iris_df.drop(labels='petal length (cm)', axis=1)
y = iris_df['petal length (cm)']
#
# X = iris_df.drop(labels='sepal length (cm)', axis=1)
# y = iris_df['sepal length (cm)']

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=111)

# linear regression-LR model
lr = LinearRegression()

# fit LR model
lr.fit(X_train, y_train)

# LR prediction
lr.predict(X_test)
y_pred = lr.predict(X_test)

# Quantative analysis - evaluate LR performance

# LR coefficients - beta/slope
print('LR beta/slope Coefficient:', lr.coef_)

# LR coefficients - alpha/slope_intercept
print('LR alpha/slope_intercept Coefficient:', lr.intercept_)

# coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: ', r2_score(y_test, y_pred))

# Model performance - Error
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))

# predict a new datapoint

# select any datapoint to predict
iris_df.loc[50]

# create a new dataframe
d = {'sepal length (cm)': [7.0],
     'sepal width (cm)': [3.2],
     'petal width (cm)': [1.4],
     'species': 0}
pred_df = pd.DataFrame(data=d)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pred_df)

# predict the new data point using LR
pred = lr.predict(pred_df)
print('Predicted Petal Length (cm)', pred[0])
print('Actual Petal Length (cm)', 4.7)

print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error([4.7], [pred])))


