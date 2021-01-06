from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris Data
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

# Concatenate the data frames
df = pd.concat([iris_df, target_df], axis=1)

# output data
print(df)

# compute the correlation coefficient for iris data set
df.corr()

# visualize iris features as a heat map
cor_eff = df.corr()
plt.figure(num='heat map', figsize=(6, 6))
sns.heatmap(cor_eff, linecolor='white', linewidths=1, annot=True)

# plot the lower half of the correlation matrix
fig, ax = plt.subplots(num='lower half', figsize=(6, 6))
# compute the correlation matrix
mask = np.zeros_like(cor_eff)

# mask = 0; display the correlation matrix, mask = 1; display the unique lower triangular values
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(cor_eff, linecolor='white', linewidths=1, mask=mask, ax=ax, annot=True)

# iris feature analysis
g = sns.pairplot(df, hue='species')

# histogram for sepal length
plt.figure(num='sepal length', figsize=(4, 4))
sepalLength = df['sepal length (cm)']
plt.hist(sepalLength, bins=10)

# histogram for sepal width
plt.figure(num='sepal width', figsize=(4, 4))
sepalWidth = df['sepal width (cm)']
plt.hist(sepalWidth, bins=10)

# histogram for petal length
plt.figure(num='petal length', figsize=(4, 4))
petalLength = df['petal length (cm)']
plt.hist(petalLength, bins=10)

# histogram for petal width
plt.figure(num='petal width', figsize=(4, 4))
petalWidth = df['petal width (cm)']
plt.hist(petalWidth, bins=10)

plt.show()

