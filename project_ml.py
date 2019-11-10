# BASIC
# -----------------------------------------------
# 1- refactor import code to use sklearn
# -----------------------------------------------
import numpy as np

# performance print results
def performance(lr):
    # Predicting the results for our test dataset
    predicted_values = lr.predict(X_test)

    print('------------------------------------------------------')

    # Printing accuracy score(mean accuracy) from 0 - 1
    print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

    # Printing the classification report
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    print('Classification Report')
    print(classification_report(y_test, predicted_values))

    # Printing the classification confusion matrix (diagonal is true)
    print('Confusion Matrix')
    print(confusion_matrix(y_test, predicted_values))

    print('Overall f1-score')
    print(f1_score(y_test, predicted_values, average="macro"))

    from sklearn import metrics
    print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
    print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
    print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")

    print('------------------------------------------------------')

# previous load method
import pandas as pd

df = pd.read_csv('data/wine.data',
                 sep=',',
                 header=0)

df.columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
              'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280 od315_of_diluted_wines',
              'proline']

# new refactored method
from sklearn.datasets import load_wine

data = load_wine()

df = pd.DataFrame(data=data['data'],
                  columns= data['feature_names'])

df.columns = data.feature_names

# -----------------------------------------------
# 2- classification regression on the wine dataset
# -----------------------------------------------
from sklearn.model_selection import train_test_split

X = data.data
y = data.target

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Logistic Regression model with fit()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

print('Classification Report for LogisticRegression')
performance(lr)

# ADVANCED
# -----------------------------------------------
# 1- using KNeighborsClassifier
# -----------------------------------------------
# Training a KNeighborsClassifier model with fit()
from sklearn.neighbors import KNeighborsClassifier

print('Classification Report for KNeighborsClassifier default n_neighbors (5)')
lr = KNeighborsClassifier()
lr.fit(X_train, y_train)
performance(lr)

print('Classification Report for KNeighborsClassifier n_neighbors = 1')
lr = KNeighborsClassifier(n_neighbors=1)
lr.fit(X_train, y_train)
performance(lr)

print('Classification Report for KNeighborsClassifier n_neighbors = 10')
lr = KNeighborsClassifier(n_neighbors=10)
lr.fit(X_train, y_train)
performance(lr)

print('Classification Report for KNeighborsClassifier n_neighbors = 20')
lr = KNeighborsClassifier(n_neighbors=20)
lr.fit(X_train, y_train)
performance(lr)

# SEEMS THAT LogisticRegression HAS THE BEST FIT

# REACH
# -----------------------------------------------
# 1- using LinearSVC
# -----------------------------------------------
# Training a LinearSVC model with fit()
print('Classification Report for LinearSVC')
from sklearn.svm import LinearSVC
lr = LinearSVC(C = 1.0)
lr.fit(X_train, y_train)
performance(lr)

# Training a SVC model with fit()
print('Classification Report for SVC')
from sklearn.svm import SVC
lr = SVC(kernel='linear', C = 1.0)
lr.fit(X_train, y_train)
performance(lr)

# Just for fun visualizing structure of dataset in 2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=3)
proj = pca.fit_transform(data.data)
plt.scatter(proj[:, 0], proj[:, 1], c=data.target, edgecolors='black')
plt.colorbar()
plt.show()

plt.close()