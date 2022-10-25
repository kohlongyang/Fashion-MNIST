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

#With edge detection
edge_detection_train = pd.read_csv('Dataset/edge-processed-fashion-mnist_train.csv')
edge_detection_test = pd.read_csv('Dataset/edge-processed-fashion-mnist_test.csv')
random.seed(3244)
edge_detection_valid = edge_detection_train.sample(frac=0.20)
edge_detection_train = edge_detection_train.drop(edge_detection_valid.index)
# print(edge_detection_train.shape)
# print(edge_detection_valid.head(3))

# with edge detection
edge_detection_Ytrain = edge_detection_train.iloc[:, 0]
edge_detection_Yvalid = edge_detection_valid.iloc[:, 0]
edge_detection_Ytest = edge_detection_test.iloc[:, 0]

edge_detection_Xtrain = edge_detection_train.loc[:,edge_detection_train.columns!='label']
edge_detection_Xvalid = edge_detection_valid.loc[:, edge_detection_valid.columns!='label']
edge_detection_Xtest = edge_detection_test.loc[:, edge_detection_test.columns!='label']

#normalise our data for edge detection, so instead of 0 to 255, become 0 to 1
edge_detection_Xtrain = np.divide(edge_detection_Xtrain, 255)
edge_detection_Xvalid = np.divide(edge_detection_Xvalid, 255)
edge_detection_Xtest = np.divide(edge_detection_Xtest, 255)
# print(edge_detection_Xtrain.head(3))
edge_detection_Xmean = edge_detection_Xtrain.mean(axis=0)
edge_detection_Xtrainfit = edge_detection_Xtrain - edge_detection_Xmean
edge_detection_Xtestfit = edge_detection_Xtest - edge_detection_Xmean

SVM = SVC(kernel = "rbf", C = 1.0, gamma="auto")

# for edge detection
# SVM.fit(edge_detection_Xtrainfit, edge_detection_Ytrain)
# edge_detection_Y_predict = SVM.predict(edge_detection_Xtestfit)
# accuracy = SVM.score(edge_detection_Xtestfit, edge_detection_Ytest)
# print('SVM Model Accuracy After Edge Detection is:' ,accuracy)
#82.76


#With data augmentation
data_augmentation_train = pd.read_csv('Dataset/augmented-fashion-mnist_train.csv')
data_augmentation_test = pd.read_csv('Dataset/fashion-mnist_test.csv')
data_augmentation_valid = data_augmentation_train.sample(frac=0.20)
data_augmentation_train = data_augmentation_train.drop(data_augmentation_valid.index)


data_augmentation_Ytrain = data_augmentation_train.iloc[:, 0]
data_augmentation_Yvalid = data_augmentation_valid.iloc[:, 0]
data_augmentation_Ytest = data_augmentation_test.iloc[:, 0]

data_augmentation_Xtrain = data_augmentation_train.loc[:,data_augmentation_train.columns!='label']
data_augmentation_Xvalid = data_augmentation_valid.loc[:, data_augmentation_valid.columns!='label']
data_augmentation_Xtest = data_augmentation_test.loc[:, data_augmentation_test.columns!='label']

#normalise our data for edge detection, so instead of 0 to 255, become 0 to 1
data_augmentation_Xtrain = np.divide(data_augmentation_Xtrain, 255)
data_augmentation_Xvalid = np.divide(data_augmentation_Xvalid, 255)
data_augmentation_Xtest = np.divide(data_augmentation_Xtest, 255)
# print(edge_detection_Xtrain.head(3))
data_augmentation_Xmean = data_augmentation_Xtrain.mean(axis=0)
data_augmentation_Xtrainfit = data_augmentation_Xtrain - data_augmentation_Xmean
data_augmentation_Xtestfit = data_augmentation_Xtest - data_augmentation_Xmean

# svm for data augmentation
SVM.fit(data_augmentation_Xtrainfit, data_augmentation_Ytrain)
data_augmentation_Y_predict = SVM.predict(data_augmentation_Xtestfit)
accuracy = SVM.score(data_augmentation_Xtestfit, data_augmentation_Ytest)
print('SVM Model Accuracy After Data Augmentation is:' ,accuracy)
#82.76
