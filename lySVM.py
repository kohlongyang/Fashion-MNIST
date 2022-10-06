import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import random

import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


#Loading of data, split into X(features) and Y(labels)
train = pd.read_csv('/Users/kohlongyang/Desktop/CS3244/fashion-mnist_train.csv')
test = pd.read_csv('/Users/kohlongyang/Desktop/CS3244/fashion-mnist_test.csv')
random.seed(3244)
valid = train.sample(frac=0.20)
#print(valid.head(3))

Ytrain = train['label']
Yvalid = valid['label']
Ytest = test['label']

Xtrain = train.loc[:,train.columns!='label']
Xvalid = valid.loc[:, valid.columns!='label']
Xtest = test.loc[:, test.columns!='label']

#actly can use sk learn to split but then train and test split alr but if use
#for validation and train maybe looks abit misleading

#normalise our data, so instead of 0 to 255, become 0 to 1
Xtrain = np.divide(Xtrain, 255)
Xvalid = np.divide(Xvalid, 255)
Xtest = np.divide(Xtest, 255)
print(Xtrain.head(3))
Xmean = Xtrain.mean(axis=0)
Xtrainfit = Xtrain - Xmean
Xtestfit = Xtest - Xmean

SVM = SVC(kernel = "rbf", C = 1.0, gamma="auto")
#https://www.geeksforgeeks.org/radial-basis-function-kernel-machine-learning/
#use radial basis function to help "draw" the decision boundary

SVM.fit(Xtrainfit, Ytrain)
Y_predict = SVM.predict(Xtestfit)
accuracy = SVM.score(Xtestfit, Ytest)
print('SVM Model Accuracy is:' ,accuracy)
#85.97%

#conf_matrix =  confusion_matrix(Ytest, Y_predict)
#PlotConfusionMatrix(conf_matrix, list(range(0,10)), normalize=True)

