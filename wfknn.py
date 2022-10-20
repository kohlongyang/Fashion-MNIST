#load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import math


#load data
df = pd.read_csv("/content/fashion-mnist_train.csv")
df_test = pd.read_csv("/content/fashion-mnist_test.csv")

#taking out labels
train_y = df.label.to_numpy()
test_y = df_test.label.to_numpy()

#remove labels from train data + normalise and scale features to [0,1]
#normalising helps to improve convergence of model

train_x = df.drop("label", axis=1).to_numpy()
train_x = train_x/255
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


#create knn model with GridSearch
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 10)
param_grid = {'n_neighbors': param_range}

knn_gscv = GridSearchCV(knn, param_grid)

gs = knn_gscv.fit(train_x, train_y)

print("best CV score = ", gs.best_score_)
print("best k= ", gs.best_params_)

# best CV score =  0.8557
# best k=  {'n_neighbors': 4}



#knn with GridSearch + PCA
gs_pca = knn_gscv.fit(reduced_trainx, train_y)

print("best CV score after PCA = ", gs.best_score_)
print("best k after PCA = ", gs.best_params_)

# best CV score after PCA =  0.8624166666666667
# best k after PCA =  {'n_neighbors': 6}

#accuracy without PCA at optimal k on test data
n = gs.best_params_['n_neighbors']
knn2 = KNeighborsClassifier(n)
knn2.fit(train_x, train_y)
y_pred = knn2.predict(test_x)
print("accuracy before PCA: ", accuracy_score(test_y,y_pred))


#accuracy with PCA at optimal k on test data
n_pca = gs_pca.best_params_['n_neighbors']
knn3 = KNeighborsClassifier(n_pca)
knn3.fit(reduced_trainx, train_y)
reduced_ypred = knn3.predict(reduced_testx)
print("accuracy after PCA: ", accuracy_score(test_y,reduced_ypred))

# accuracy before PCA:  0.8606
# accuracy after PCA:  0.8697

knn3 = KNeighborsClassifier(4)
knn3.fit(reduced_trainx, train_y)
reduced_ypred = knn3.predict(reduced_testx)
print("accuracy after PCA: ", accuracy_score(test_y,reduced_ypred))

"""Graph of CV Score against K values for kNN"""

#plot graph of cv score against k value
cv_results = pd.DataFrame(gs.cv_results_)
grid_mean_scores = cv_results['mean_test_score']
k_range = list(range(1, 10))


plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

cv_results_pca = pd.DataFrame(gs_pca.cv_results_)
grid_mean_scores_pca = cv_results_pca['mean_test_score']

plt.plot(k_range, grid_mean_scores_pca)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


#manhattan distance
#use gridsearchCV to find optimal k

knn_man = KNeighborsClassifier(metric="manhattan")

#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 10)
param_grid = {'n_neighbors': param_range}
#use gridsearch to test all values for n_neighbors
knn_man_gscv = GridSearchCV(knn_man, param_grid)
#fit model to data
gs = knn_man_gscv.fit(train_x, train_y)

#get best performing models
print("best accuracy= ", gs.best_score_)
print("best k= ", gs.best_params_)

# best accuracy=  0.7525000000000001
# best k=  {'n_neighbors': 3}

knn_man = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
knn_man.fit(train_x, train_y)
y_pred_man = knn_man.predict(test_x)
print("accuracy using Manhattan distance: ", accuracy_score(test_y,y_pred_man))

#accuracy using Manhattan distance:  0.8664
