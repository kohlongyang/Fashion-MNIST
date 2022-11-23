import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import random
import sklearn
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

#load data
df = pd.read_csv("/content/drive/MyDrive/newest-edge-processed-fashion-mnist_train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/newest-edge-processed-fashion-mnist_test.csv")
df = df.dropna()
df_test = df_test.dropna()

random.seed(0)
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
pca = PCA(n_components = 0.90)
pca.fit(train_x)
reduced_trainx = pca.transform(train_x)
reduced_testx = pca.transform(test_x)

#getting how many components we reduced to
pca.n_components_
#431

#plot graph of finding principal components needed in PCA to explain 90% of variance
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

plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.85, '90% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.savefig("PCA-edge detection")

#create knn model with GridSearch
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_range = np.arange(1, 10)
param_grid = {'n_neighbors': param_range}

knn_gscv = GridSearchCV(knn, param_grid)

gs = knn_gscv.fit(train_x, train_y)

print("best CV score = ", gs.best_score_)
print("best k= ", gs.best_params_)

# best CV score =  0.8037171239318098
# best k=  {'n_neighbors': 5}


#knn with GridSearch + PCA
gs_pca = knn_gscv.fit(reduced_trainx, train_y)

print("best CV score after PCA = ", gs.best_score_)
print("best k after PCA = ", gs.best_params_)

# best CV score after PCA =  0.8197618237505111
# best k after PCA =  {'n_neighbors': 6}

#accuracy without PCA at optimal k on test data
n = gs.best_params_['n_neighbors']
knn2 = KNeighborsClassifier(n)
knn2.fit(train_x, train_y)
y_pred = knn2.predict(test_x)
print("accuracy before PCA: ", accuracy_score(test_y,y_pred))

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cm = confusion_matrix(test_y, y_pred)
plot = sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
fig = plot.get_figure()
fig_name = "KNN without PCA"
fig.savefig(fig_name, bbox_inches='tight')
plt.clf()
cr = classification_report(test_y, y_pred)
print(cr)

#accuracy with PCA at optimal k on test data
n_pca = gs_pca.best_params_['n_neighbors']
knn3 = KNeighborsClassifier(n_pca)
knn3.fit(reduced_trainx, train_y)
reduced_ypred = knn3.predict(reduced_testx)
print("accuracy after PCA: ", accuracy_score(test_y,reduced_ypred))

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cm = confusion_matrix(test_y, reduced_ypred)
plot = sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
fig = plot.get_figure()
fig_name = "KNN with PCA"
fig.savefig(fig_name, bbox_inches='tight')
plt.clf()
cr = classification_report(test_y, reduced_ypred)
print(cr)

# accuracy before PCA:  0.8061
# accuracy after PCA:  0.8237

# without pca
# finding out best C for k=6
penalties = np.logspace(-2,3, num=10)
#using k=6
mcv_scores = []
for C in penalties:
    curr_svm = SVC(kernel='rbf', C=C)
    fold = KFold(n_splits=6, random_state=1, shuffle=True)
    # Cross validation 6-Fold scores
    mean_crossval = np.mean(cross_val_score(curr_svm, train_x, train_y, cv=fold))
    mcv_scores.append(mean_crossval)
    print("Without PCA")
    print("On C=", C, "\tMCV=", mean_crossval)
# with pca c = 6, MCV = 87.3%

# finding out best C for k=6
penalties = np.logspace(-2,3, num=10)
#using k=3
mcv_scores = []
for C in penalties:
    curr_svm = SVC(kernel='rbf', C=C)
    fold = KFold(n_splits=3, random_state=1, shuffle=True)
    # Cross validation 3-Fold scores
    mean_crossval = np.mean(cross_val_score(curr_svm, reduced_trainx, train_y, cv=fold))
    mcv_scores.append(mean_crossval)
    print("With PCA")
    print("On C=", C, "\tMCV=", mean_crossval)

#accuracy without PCA at optimal c on test data, c = 6 without PCA
svm = SVC(kernel='rbf', C= 6)
svm.fit(train_x, train_y)
y_pred = svm.predict(test_x)
print("accuracy before PCA: ", accuracy_score(test_y,y_pred))

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cm = confusion_matrix(test_y, y_pred)
plot = sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
fig = plot.get_figure()
fig_name = "SVM without PCA"
fig.savefig(fig_name, bbox_inches='tight')
plt.clf()
cr = classification_report(test_y, y_pred)
print(cr)

#accuracy with PCA at optimal c on test data On C= 21.544346900318846 	MCV= 0.870381323251673
svm = SVC(kernel='rbf', C= 21.54)
svm.fit(reduced_trainx, train_y)
reduced_ypred = svm.predict(reduced_testx)
print("accuracy before PCA: ", accuracy_score(test_y,reduced_ypred))

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

cm = confusion_matrix(test_y, reduced_ypred)
plot = sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
fig = plot.get_figure()
fig_name = "SVM with PCA"
fig.savefig(fig_name, bbox_inches='tight')
plt.clf()
cr = classification_report(test_y,reduced_ypred)
print(cr)