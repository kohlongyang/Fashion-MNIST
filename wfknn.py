

#load packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
import pandas as pd

#data preprocessing

#load data
df = pd.read_csv("/content/fashion-mnist_train.csv")
df_test = pd.read_csv("/content/fashion-mnist_test.csv")

df.head() #first column is label

#check dimensions
print(df.shape)

#select small set from training set to model test to test out
trial = df.iloc[:1000,]

#taking out labels
trial_y = trial.label

#remove labels from train data
#normalise and scale features to [0,1] -> to improve convergence/learning of model 
trial_x = trial.drop("label", axis=1)/255.0
#trial_train_x.head()

#show example of 1 image
plt.imshow(trial_x.iloc[0,:].values.reshape([28,28])) ; trial_y.iloc[0]

# can split the data like using sklearn but only train and test data, no validation data

# from sklearn.model_selection import train_test_split
# res = train_test_split(data, labels, 
#                        train_size=0.8,
#                        test_size=0.2,
#                        random_state=1)

# train_data, test_data, train_labels, test_labels = res

#splitting trial set into train, validation and test data

np.random.seed(3244)
train_indices = np.random.choice(len(trial_x), round(len(trial_x)*0.8), replace=False)
valid_indices = np.random.choice(list(set(range(len(trial_x))) - set(train_indices)), round(len(train_indices)*0.2), replace=False)
test_indices = np.array(list(set(range(len(trial_x)))- set(train_indices) - set(valid_indices)))

#split train set into train and validation dataset
# train_indices = np.random.choice(len(train_indices), round(len(train_indices)*0.8), replace=False)
# valid_indices = np.array(list(set(range(len(train_indices))) - set(train_indices)))

train_x = trial_x.iloc[train_indices, :]
train_y = trial_y.iloc[train_indices]

valid_x = trial_x.iloc[valid_indices, :]
valid_y = trial_y.iloc[valid_indices]

test_x = trial_x.iloc[test_indices, :]
test_y = trial_y.iloc[test_indices]

#checking if there's overlap in indices/data
np.in1d(train_indices, test_indices).any()

## knn models

#load packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score
import math

#use euclidean distance

KNN_model_4 = KNeighborsClassifier(n_neighbors=4).fit(train_x, train_y)


#check training accuracy
knn_4_trainacc = (KNN_model_4.predict(train_x) == train_y).mean()
print(knn_4_trainacc)
#check validation accuracy
knn_4_validacc = (KNN_model_4.predict(valid_x) == valid_y).mean()
print(knn_4_validacc)

# 0.8325
# 0.7

#create knn model with k=5
KNN_model_5 = KNeighborsClassifier().fit(train_x, train_y)

#getting accuracies
confusion_matrix(KNN_model_5.predict(train_x), train_y)

knn_5_trainacc = (KNN_model_5.predict(train_x) == train_y).mean()
print("k=5, train accuracy= ", knn_5_trainacc)

knn_5_validacc = (KNN_model_5.predict(valid_x) == valid_y).mean()
print("k=5, validation accuracy= ", knn_5_validacc)

# k=5, train accuracy=  0.80625
# k=5, validation accuracy=  0.70625

KNN_model_6 = KNeighborsClassifier(n_neighbors=6).fit(train_x, train_y)


#check training accuracy
knn_6_trainacc = (KNN_model_6.predict(train_x) == train_y).mean()
print(knn_6_trainacc)
#check validation accuracy
knn_6_validacc = (KNN_model_6.predict(valid_x) == valid_y).mean()
print(knn_6_validacc)

# 0.80125
# 0.70625

KNN_model_8 = KNeighborsClassifier(n_neighbors=8).fit(train_x, train_y)


#check training accuracy
knn_8_trainacc = (KNN_model_8.predict(train_x) == train_y).mean()
print(knn_8_trainacc)
#check validation accuracy
knn_8_validacc = (KNN_model_8.predict(valid_x) == valid_y).mean()
print(knn_8_validacc)

# 0.7925
# 0.7375

#use gridsearchcv

#can use sklearn GridSearchCV to find most optimal 
from sklearn.model_selection import GridSearchCV

#create new a knn model
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

#Reference: 

#https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

#https://medium.com/luca-chuangs-bapm-notes/using-grid-search-to-tune-our-machine-learning-models-f725b78c3ede


#manhattan distance


#creating and fitting knn model with manhattan distance metric

#k=5
knn_man_5 = KNeighborsClassifier(n_neighbors=5, metric="manhattan").fit(train_x, train_y)

#getting accuracies
#check training accuracy
knn_man_5_t = (knn_man_5.predict(train_x) == train_y).mean()
print(knn_man_5_t)
#check validation accuracy
knn_man_5_v = (knn_man_5.predict(valid_x) == valid_y).mean()
print(knn_man_5_v)

# 0.82125
# 0.7375

knn_man_3 = KNeighborsClassifier(n_neighbors=3, metric="manhattan").fit(train_x, train_y)

#getting accuracies
#check training accuracy
knn_man_3_t = (knn_man_3.predict(train_x) == train_y).mean()
print(knn_man_3_t)
#check validation accuracy
knn_man_3_v = (knn_man_3.predict(valid_x) == valid_y).mean()
print(knn_man_3_v)

# 0.85375
# 0.7



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

# best accuracy=  0.7525000000000001
# best k=  {'n_neighbors': 3}