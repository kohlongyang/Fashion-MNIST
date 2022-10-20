#load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


#data preprocessing

#load data
df = pd.read_csv("fashion-mnist_train.csv")
df_test = pd.read_csv("fashion-mnist_test.csv")

#taking out labels
train_y = df.label.to_numpy()
test_y = df_test.label.to_numpy()

#remove labels from train data
#normalise and scale features to [0,1] -> to improve convergence/learning of model 
train_x = df.drop("label", axis=1).to_numpy()
train_x =train_x/255
test_x = df_test.drop("label", axis=1).to_numpy()
test_x = test_x/255


#dimension reduction using PCA
pca = PCA(n_components = 0.95)
pca.fit(train_x)
reduced_trainx = pca.transform(train_x)
reduced_testx = pca.transform(test_x)

#getting how many components we reduced to
pca.n_components_
#187

#plot graph of finding principal components needed in PCA to explain 95% of variance
pca1 = PCA().fit(train_x)

plt.rcParams["figure.figsize"] = (36,18)

fig, ax = plt.subplots()
xi = np.arange(0, 784, step=1)
y = np.cumsum(pca1.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

#knn + GridSearchCV
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 10)
param_grid = {'n_neighbors': param_range}
knn_gscv = GridSearchCV(knn2, param_grid)
gs = knn_gscv.fit(train_x, train_y)

print("best accuracy= ", gs.best_score_)
print("best k= ", gs.best_params_)

# best accuracy=  0.8557
# best k=  {'n_neighbors': 4}

#knn + GridSearch + PCA
gs_pca = knn_gscv.fit(reduced_trainx, train_y)
print("best CV score after PCA = ", gs.best_score_)
print("best k after PCA = ", gs.best_params_)

#accuracy w/o PCA at optimal k on test data
n = gs.best_params_['n_neighbors']
knn2 = KNeighborsClassifier(n)
knn2.fit(train_x, train_y)
y_pred = knn2.predict(test_x)
print("accuracy before PCA: ", accuracy_score(test_y,y_pred))

#accuracy w PCA at optimal k on test data
n_pca = gs_pca.best_params_['n_neighbors']
knn3 = KNeighborsClassifier(n_pca)
knn3.fit(reduced_trainx, train_y)
reduced_ypred = knn3.predict(reduced_testx)
print("accuracy after PCA: ", accuracy_score(test_y,reduced_ypred))

# accuracy before PCA:  0.8606
# accuracy after PCA:  0.8697



#manhattan distance, use gridsearchCV to find optimal k

knn3 = KNeighborsClassifier(metric="manhattan")
#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 10)
param_grid = {'n_neighbors': param_range}
#use gridsearch to test all values for n_neighbors
knn_man_gscv = GridSearchCV(knn3, param_grid)
#fit model to data
gs = knn_man_gscv.fit(train_x, train_y)

#get best performing models
print("best accuracy= ", gs.best_score_)
print("best k= ", gs.best_params_)