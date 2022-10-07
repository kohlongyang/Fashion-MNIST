import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import random

import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


#Loading of data, split into X(features) and Y(labels)
train = pd.read_csv('/Users/kohlongyang/Desktop/fashion-mnist_train.csv')
test = pd.read_csv('/Users/kohlongyang/Desktop/fashion-mnist_test.csv')
random.seed(3244)
valid = train.sample(frac=0.20)
train = train.drop(valid.index)
#print(train.shape)
#print(valid.head(3))

Ytrain = train.iloc[:, 0]
Yvalid = valid.iloc[:, 0]
Ytest = test.iloc[:, 0]

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
#85.57%

conf_matrix =  confusion_matrix(Ytest, Y_predict)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()
#can see that classes 1,3,5,8,9 are classified well

#not using the original train because i want to keep the original data
XYtrain = pd.concat([pd.DataFrame(data=Ytrain), pd.DataFrame(data=Xtrain)]
                , axis=1)

#small samples of data
Xsmall = pd.concat([XYtrain[XYtrain['label']==i].iloc[:1000, 1:785] for i in range(0,10)], axis=0).values
Ysmall = pd.concat([XYtrain[XYtrain['label']==i].iloc[:1000, 0] for i in range(0,10)], axis=0).values

#need to shuffle because above method makes it in order of 0-9
Xsmall, Ysmall = shuffle(Xsmall, Ysmall)

#setting my penalties in a logspace, from 10^-2 to 10^3
penalties = np.logspace(-2,3, num=10)

#using k=3
mcv_scores = []
for C in penalties:
    curr_svm = SVC(kernel='rbf', C=C)
    fold = KFold(n_splits=3, random_state=1, shuffle=True)
    # Cross validation 3-Fold scores        
    mean_crossval = np.mean(cross_val_score(curr_svm, Xsmall, Ysmall, cv=fold))
    mcv_scores.append(mean_crossval)
    print("On C=", C, "\tMCV=", mean_crossval)
#highest MCV of 87.01% when C = 21.544

#using k=10 --> highest MCV of 87.79% when C = 5.994