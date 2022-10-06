#load packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
import math

#data preprocessing

#load data
df = pd.read_csv("/content/fashion-mnist_train.csv")
df_test = pd.read_csv("/content/fashion-mnist_test.csv")

#taking out labels
train_y = df.label.to_numpy()
test_y = df_test.label.to_numpy()

#remove labels from train data
#normalise and scale features to [0,1] -> to improve convergence/learning of model 
train_x = df.drop("label", axis=1).to_numpy()
train_x =train_x/255
test_x = df_test.drop("label", axis=1).to_numpy()
test_x = test_x/255

#show example of 1 image
plt.imshow(train_x.iloc[0,:].values.reshape([28,28])) ; train_y.iloc[0]


#knn models
#use euclidean distance
#use gridsearchcv to find optimal k
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 25)
param_grid = {'n_neighbors': param_range}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid)

#fit model to data
gs = knn_gscv.fit(train_x, train_y)

#get best performing models
print("best accuracy= ", gs.best_score_)
print("best k= ", gs.best_params_)

# best accuracy=  0.8557
# best k=  {'n_neighbors': 4}



#manhattan distance
#use gridsearchCV to find optimal k

knn3 = KNeighborsClassifier(metric="manhattan")
#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 25)
param_grid = {'n_neighbors': param_range}
#use gridsearch to test all values for n_neighbors
knn_man_gscv = GridSearchCV(knn3, param_grid)
#fit model to data
gs = knn_man_gscv.fit(train_x, train_y)

#get best performing models
print("best accuracy= ", gs.best_score_)
print("best k= ", gs.best_params_)

