import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
#from lySVM import Y_testpredict

import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from __future__ import print_function
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn import decomposition
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense

#Loading of data, split into X(features) and Y(labels)
train = pd.read_csv('/Users/kohlongyang/Desktop/fashion-mnist_train.csv')
test = pd.read_csv('/Users/kohlongyang/Desktop/fashion-mnist_test.csv')

random.seed(3244)
valid = pd.read_csv('/Users/kohlongyang/Desktop/validation.csv', index_col=0)
train = pd.read_csv('/Users/kohlongyang/Desktop/training.csv', index_col=0)
#print(train.shape)
#print(valid.head(3))


Ytrain = train.iloc[:, 0]
Yvalid = valid.iloc[:, 0]
Ytest = test.iloc[:, 0]

Xtrain = train.loc[:,train.columns!='label']
Xvalid = valid.loc[:, valid.columns!='label']
Xtest = test.loc[:, test.columns!='label']

Xtrain = np.divide(Xtrain, 255)
Xvalid = np.divide(Xvalid, 255)
Xtest = np.divide(Xtest, 255)

#first scree plot shows that the number of optimum comps is within the range of 1 to 30
covar_matrix = PCA(n_components=784)
covar_matrix.fit(Xtrain)
covar_matrix_values = np.arange(covar_matrix.n_components_)+1
plt.style.context('seaborn-whitegrid')
plt.plot(covar_matrix_values,covar_matrix.explained_variance_ratio_,'o-',linewidth=2,color='blue')
plt.title('Scree Plot')
plt.xlabel('PC')
plt.ylabel('Variance Explained')

#reduce the dimensions further and double confirm
covar_matrix = PCA(n_components=100)
covar_matrix.fit(Xtrain)
covar_matrix_values = np.arange(covar_matrix.n_components_)+1
plt.style.context('seaborn-whitegrid')
plt.plot(covar_matrix_values,covar_matrix.explained_variance_ratio_,'o-',linewidth=2,color='blue')
plt.title('Scree Plot')
plt.xlabel('PC')
plt.ylabel('Variance Explained')
#90% of variance explained happens at 84 components

#a for loop to see where the accuracy starts to get stagnant/insignificant increase in accuracy
n_components = np.arange(1,85) 
for n in n_components: 
    pca = PCA(n_components=n, random_state=42)
    Xtrainpca = pca.fit_transform(Xtrain)
    Xvalidpca = pca.transform(Xvalid)
    Xtestpca = pca.transform(Xtest)

#Using GSCV to find best k
    knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
    param_range = np.arange(1, 10)
    param_grid = {'n_neighbors': param_range}

#use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn, param_grid)

#fit model to data
    gs = knn_gscv.fit(Xtrainpca, Ytrain)

#get best performing models
    print("best accuracy= ", gs.best_score_)
    print("best k= ", gs.best_params_)
# best accuracy=  0.8363125, at n_components = 16, best k = 6 
# best accuracy = 0.8602083, at n_components = 84, best k = 8


#finding out best C for k=8
penalties = np.logspace(-2,3, num=10)
#using k=8
mcv_scores = []
for C in penalties:
    curr_svm = SVC(kernel='rbf', C=C)
    fold = KFold(n_splits=8, random_state=1, shuffle=True)
    # Cross validation 8-Fold scores        
    mean_crossval = np.mean(cross_val_score(curr_svm, Xtrainpca, Ytrain, cv=fold))
    mcv_scores.append(mean_crossval)
    print("On C=", C, "\tMCV=", mean_crossval)
# highest MCV of 89.93% when C = 21.54 and k = 8


pca = PCA(n_components=84, random_state=42)
Xtrainpca = pca.fit_transform(Xtrain)
Xvalidpca = pca.transform(Xvalid)
Xtestpca = pca.transform(Xtest)
SVM = SVC(kernel = "rbf", C = 21.54, gamma="auto")
# #https://www.geeksforgeeks.org/radial-basis-function-kernel-machine-learning/
# #use radial basis function to help "draw" the decision boundary
SVM.fit(Xtrainpca, Ytrain)
Y_predict = SVM.predict(Xtrainpca)
train_acc = metrics.accuracy_score(Ytrain,Y_predict)
#0.9575625
Y_validpredict = SVM.predict(Xvalidpca)
valid_acc = metrics.accuracy_score(Yvalid,Y_validpredict)
#0.88508
Y_testpredict = SVM.predict(Xtestpca)
test_acc = metrics.accuracy_score(Ytest,Y_testpredict)
#89.87%
results = metrics.f1_score(Ytest,Y_testpredict, average = 'micro')
#0.8987
results = metrics.f1_score(Ytest,Y_testpredict, average = 'macro')
#0.898378
label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
testresult = metrics.classification_report(Ytest, Y_testpredict, target_names=label)
print(testresult)

conf_matrix =  confusion_matrix(Ytest, Y_testpredict)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()

#we can see that besides 4 and 6, they are mostly well classified
#also can see that 0 and 6 got significant number of wrong predictions because 
# 0 is tshirt/top then 6 is shirt, then 2 and 4 which is pullover and coat


#Extra Data Visualisation
#According to Scree Plot, elbow is at n=3. 
features = ['pixel' + str(i+1) for i in range(Xtrain.shape[1])]
vispca_df = pd.DataFrame(Xtrain, columns=features)
label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
target = train[['label']].iloc[:,:]
vispca_df['label'] = target['label']

# Create an empty list which will save all meaningful labels
results = []
# Loop through all label
for i in range(vispca_df.shape[0]):
    # Extract the label for comparison
    if vispca_df['label'][i] == 0:
        # Save meaningful label to the results
        results.append('T-shirt/top')
    # Following the same code pattern as the one above
    elif vispca_df['label'][i] == 1:
        results.append('Trouser')
    elif vispca_df['label'][i] == 2:
        results.append('Pullover')
    elif vispca_df['label'][i] == 3:
        results.append('Dress')
    elif vispca_df['label'][i] == 4:
        results.append('Coat')
    elif vispca_df['label'][i] == 5:
        results.append('Sandal')
    elif vispca_df['label'][i] == 6:
        results.append('Shirt')
    elif vispca_df['label'][i] == 7:
        results.append('Sneaker')
    elif vispca_df['label'][i] == 8:
        results.append('Bag')
    elif vispca_df['label'][i] == 9:
        results.append('Ankle boot')
    else:
        print("The dataset contains an unexpected label {}".format(vispca_df['label'][i]))

# Create a new column named result which has all meaningful results        
vispca_df['result'] = results


pca3 = PCA(n_components=3)
pca3_result = pca3.fit_transform(vispca_df[features].values)
print('Explained variation per principal component: {}'.format(pca3.explained_variance_ratio_))

vispca_df['First Dimension'] = pca3_result[:,0]
vispca_df['Second Dimension'] = pca3_result[:,1] 
vispca_df['Third Dimension'] = pca3_result[:,2]

plt.style.use('fivethirtyeight')
#Set figure size
fig, axarr = plt.subplots(1, 3, figsize=(20, 5))
# use seaborn heatmap to visualize the first three pca components
sns.heatmap(pca3.components_[0, :].reshape(28, 28), ax=axarr[0], cmap=plt.cm.binary)
sns.heatmap(pca3.components_[1, :].reshape(28, 28), ax=axarr[1], cmap=plt.cm.binary)
sns.heatmap(pca3.components_[2, :].reshape(28, 28), ax=axarr[2], cmap=plt.cm.binary)
# Set picture title to explained variance
axarr[0].set_title(
    "{0:.2f}% Explained Variance".format(pca3.explained_variance_ratio_[0]*100), fontsize=14)
axarr[1].set_title(
    "{0:.2f}% Explained Variance".format(pca3.explained_variance_ratio_[1]*100), fontsize=14)
axarr[2].set_title(
    "{0:.2f}% Explained Variance".format(pca3.explained_variance_ratio_[2]*100), fontsize=14)
# Add picture title
plt.suptitle('3-Component PCA')

#comparing PCA vs AE
input_img = Input(shape = (784,))

encoded = Dense(32, activation="relu")(input_img)
encoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(decoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer="rmsprop", loss="binary_crossentropy")

#training the model
hist = autoencoder.fit(Xtrain, 
                       Xtrain,
                       epochs=250, 
                       batch_size=256, 
                       shuffle=True,
                       validation_data=(Xtrain, Xtrain))

encoder = Model(input_img, encoded)
encoded_img = encoder.predict(Xtest)

#Getting original image
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(new[25].reshape(28,28), cmap="gist_yarg") #new is Xtest df converted to array
plt.title("Original Image", color = "green")
plt.axis("off")

#Image resulting from encoder
plt.subplot(2, 2, 2)
plt.imshow(encoded_img[25].reshape(4,4), cmap="gist_yarg")
plt.title("Image Resulting from Encoder", color = "Darkred")
plt.axis("off")
plt.show()

decoded_imgs = autoencoder.predict(Xtest) 
plt.figure(figsize=(20, 5))
for i in range(10):
    
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gist_yarg")
    plt.title("(Original)\nT-shirt/top" if Y_test[i] == 0 
              else "(Original)\nTrouser" if Y_test[i] == 1
              else "(Original)\nPullover" if Y_test[i] == 2
              else "(Original)\nDress" if Y_test[i] == 3
              else "(Original)\nCoat" if Y_test[i] == 4
              else "(Original)\nSandal" if Y_test[i] == 5
              else "(Original)\nShirt" if Y_test[i] == 6
              else "(Original)\nSneaker" if Y_test[i] == 7
              else "(Original)\nBag" if Y_test[i] == 8
              else "(Original)\nAnkle boot", size = 13, color = "darkred")
    plt.axis("off")

    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.title("(Reconstructed)\nT-shirt/top" if Y_test[i] == 0 
              else "(Reconstructed)\nTrouser" if Y_test[i] == 1
              else "(Reconstructed)\nPullover" if Y_test[i] == 2
              else "(Reconstructed)\nDress" if Y_test[i] == 3
              else "(Reconstructed)\nCoat" if Y_test[i] == 4
              else "(Reconstructed)\nSandal" if Y_test[i] == 5
              else "(Reconstructed)\nShirt" if Y_test[i] == 6
              else "(Reconstructed)\nSneaker" if Y_test[i] == 7
              else "(Reconstructed)\nBag" if Y_test[i] == 8
              else "(Reconstructed)\nAnkle boot", size = 13, color = "green")
    plt.axis("off")

plt.show()

# visualising PCA vs AE
X_reduced = pca.fit_transform(Xtest)
X_recovered = pca.inverse_transform(X_reduced)
plt.imshow(new[25].reshape(28, 28), cmap="binary") #original image
plt.imshow(X_recovered[25].reshape(28, 28), cmap="binary") #compressed image
plt.imshow(encoded_img[25].reshape(4,4), cmap="gist_yarg")
plt.imshow(decoded_imgs[25].reshape(28, 28)) #AE image