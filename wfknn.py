#load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import math

##data processing

#load data
df_train = pd.read_csv("/content/drive/My Drive/fashion-mnist_train.csv")
df_test = pd.read_csv("/content/drive/My Drive/fashion-mnist_test.csv")

#training-validation split
df, df_valid = train_test_split(df_train, test_size=0.2, random_state=42)

#taking out labels
train_y = df.label.to_numpy()
valid_y = df_valid.label.to_numpy()
test_y = df_test.label.to_numpy()

#remove labels from train data + normalise and scale features

train_x = df.drop("label", axis=1).to_numpy()
train_x = train_x/255
valid_x = df_valid.drop("label", axis=1).to_numpy()
valid_x = valid_x/255

test_x = df_test.drop("label", axis=1).to_numpy()
test_x = test_x/255

# np.isnan(train_x).any()






##dimension reduction using PCA

pca = PCA(n_components = 0.90)
pca.fit(train_x)
reduced_trainx = pca.transform(train_x)
reduced_validx = pca.transform(valid_x)
reduced_testx = pca.transform(test_x)

#getting how many components we reduced to
pca.n_components_
#84


#plot graph of finding principal components needed in PCA to explain 95% of variance
pca1 = PCA().fit(train_x)

plt.rcParams["figure.figsize"] = (18,9)

fig, ax = plt.subplots()
xi = np.arange(0, 784, step=1)
y = np.cumsum(pca1.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='blue')

plt.xlabel('Number of Components')
#plt.xticks(np.arange(0, 784, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(2.0, 0.90, '90% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()







#knn model with GridSearchCV

knn = KNeighborsClassifier()

#first sweep
param_range1 = [5, 50, 100, 250]
param_grid1 = {'n_neighbors': param_range1}

#second sweep
param_range2 = np.arange(1, 10)
param_grid2 = {'n_neighbors': param_range2}

knn_gscv = GridSearchCV(knn, param_grid2)
gs = knn_gscv.fit(train_x, train_y)

print("best CV score = ", gs.best_score_)
print("best k= ", gs.best_params_)



#knn with GridSearch + PCA

gs_pca = knn_gscv.fit(reduced_trainx, train_y)

print("best CV score after PCA = ", gs.best_score_)
print("best k after PCA = ", gs.best_params_)

#plot graph of cv score against k value
cv_results = pd.DataFrame(gs.cv_results_)
grid_mean_scores = cv_results['mean_test_score']
k_range = list(range(1, 9))


plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')




##confusion matrix

ls = np.arange(0,10,1)

#pre-PCA on validation
cm = confusion_matrix(valid_y, valid_ypred, labels=ls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#post-PCA on validation
cm = confusion_matrix(valid_y, reduced_valid_ypred, labels=ls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#pre-PCA on test
test_ypred = knn2.predict(test_x)

cm_test = confusion_matrix(test_y, test_ypred, labels=ls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()

#post-PCA on test
reduced_test_ypred = knn3.predict(reduced_testx)

cm_reduced_test = confusion_matrix(test_y, reduced_test_ypred, labels=ls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_reduced_test)
disp.plot()
plt.show()






## classification report
#pre-PCA on validation
print(classification_report(valid_y, valid_ypred, labels=ls))

#post-PCA on validation
print(classification_report(valid_y, reduced_valid_ypred, labels=ls))

"""Test"""

#pre-PCA on test
print(classification_report(test_y, actual_ypred, labels=ls))

#post-PCA on test
print(classification_report(test_y, reduced_actual_ypred, labels=ls))







#finding misclassified instances

n = len(test_y)-1
wrong = []

for i in range(n):
  if test_ypred[i] != test_y[i]:
    w = [i, test_ypred[i], test_y[i]]
    wrong.append(w)

wrong_pca = []

for i in range(n):
  if reduced_test_ypred[i] != test_y[i]:
    p = [i, reduced_test_ypred[i], test_y[i]]
    wrong_pca.append(p)

#printing misclassified instance
ls = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

no, predicted, actual = wrong_pca[25]
print("predicted: ",predicted, ls[predicted])
print("actual: ", actual, ls[actual])

plt.imshow(test_x[no,:].reshape([28,28]), cmap=plt.cm.binary) ; test_y[no]
