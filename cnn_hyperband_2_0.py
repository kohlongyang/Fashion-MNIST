
import keras
import sklearn
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv("/content/drive/MyDrive/CS3244 Project CNN/training.csv")
validation = pd.read_csv("/content/drive/MyDrive/CS3244 Project CNN/validation.csv")
test = pd.read_csv("/content/drive/MyDrive/CS3244 Project CNN/fashion-mnist_test.csv")

train_X = train.iloc[:, 2:].to_numpy()
train_Y = train.iloc[:, 1].to_numpy()
valid_X = validation.iloc[:, 2:].to_numpy()
valid_Y = validation.iloc[:, 1].to_numpy()
test_X = test.iloc[:, 1:].to_numpy()
test_Y = test.iloc[:, 0].to_numpy()

## Data Exploration and Pre-processing

train_X = train_X.reshape(-1, 28, 28)
valid_X = valid_X.reshape(-1, 28, 28)
test_X = test_X.reshape(-1, 28, 28)

train_X.shape, train_Y.shape

valid_X.shape, valid_Y.shape

test_X.shape, test_Y.shape

plt.figure(figsize=[10,10])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

train_X = train_X.reshape(-1, 28,28, 1)
valid_X = valid_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, valid_X.shape, test_X.shape

train_X = train_X.astype('float32')
valid_X = valid_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
valid_X = valid_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
valid_Y_one_hot = to_categorical(valid_Y)
test_Y_one_hot = to_categorical(test_Y)

# CNN model

from keras.models import Sequential,Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU

!pip install keras_tuner --upgrade
import keras_tuner

def build_model(hp):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(hp.Choice('units1.1', [16, 32]), kernel_size=(hp.Choice('units1.2', [2, 3])),activation=hp.Choice("activation1", ["relu", "sigmoid", "tanh"]),input_shape=(28,28,1),padding='same'))
    pool_size1 = hp.Choice('units1.3', [1, 2])
    cnn_model.add(MaxPooling2D((pool_size1, pool_size1),padding='same'))
    dropout_ratio1 = hp.Float("dropout_ratio1", min_value=0.1, max_value=0.25, step=0.01)
    if hp.Boolean("dropout1"):
        cnn_model.add(Dropout(rate=dropout_ratio1))
      
    cnn_model.add(Conv2D(hp.Choice('units2.1', [32, 64]), kernel_size=(hp.Choice('units2.2', [2, 3])),activation=hp.Choice("activation2", ["relu", "sigmoid", "tanh"]),input_shape=(28,28,1),padding='same'))
    pool_size2 = hp.Choice('units2.3', [1, 2])
    cnn_model.add(MaxPooling2D((pool_size2, pool_size2),padding='same'))
    dropout_ratio2 = hp.Float("dropout_ratio2", min_value=0.1, max_value=0.25, step=0.01)
    if hp.Boolean("dropout2"):
        cnn_model.add(Dropout(rate=dropout_ratio2))
      
    cnn_model.add(Conv2D(hp.Choice('units3.1', [64, 128]), kernel_size=(hp.Choice('units3.2', [2, 3])),activation=hp.Choice("activation3", ["relu", "sigmoid", "tanh"]),input_shape=(28,28,1),padding='same'))
    pool_size3 = hp.Choice('units3.3', [1, 2])
    cnn_model.add(MaxPooling2D((pool_size3, pool_size3),padding='same'))
    dropout_ratio3 = hp.Float("dropout_ratio3", min_value=0.1, max_value=0.25, step=0.01)
    if hp.Boolean("dropout3"):
        cnn_model.add(Dropout(rate=dropout_ratio3))
    
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation = hp.Choice("activation4", ["relu", "sigmoid", "tanh"])))
    dropout_ratio4 = hp.Float("dropout_ratio4", min_value=0.1, max_value=0.25, step=0.01)
    if hp.Boolean("dropout4"):
        cnn_model.add(Dropout(rate=dropout_ratio4))
    
    cnn_model.add(Dense(10, activation="softmax"))
    adam_lr = hp.Float("lr", min_value=1e-4, max_value=1e-3, sampling="log")
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(adam_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return cnn_model

## Keras Tuner

tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_accuracy",
    max_epochs=35,
    hyperband_iterations = 1,
    seed = 10,
    overwrite = False,
    directory = "/content/drive/MyDrive/CS3244 Project CNN",
    project_name = "Hyperband_Tuner"
)

#tuner.search(train_X, train_Y_one_hot, epochs=40, validation_data=(valid_X, valid_Y_one_hot))

### Hyperparameter tuning results

tuner.results_summary()

# Evaluation (Validation data)

!pip install visualkeras --upgrade
import visualkeras

best_model = tuner.get_best_models(1)[0]

visualkeras.layered_view(best_model, legend = True)

keras.utils.plot_model(best_model, to_file="model.png", show_shapes=True, show_dtype=False, show_layer_names=True, rankdir="TB", 
                       expand_nested=True, dpi=96, layer_range=None, show_layer_activations=True)

valid_predicted_classes = best_model.predict(valid_X)
valid_predicted_classes = np.argmax(np.round(valid_predicted_classes),axis=1)

valid_eval = best_model.evaluate(valid_X, valid_Y_one_hot, verbose=1)

print('Valid loss:', valid_eval[0])
print('Valid accuracy:', valid_eval[1])

correct = np.where(valid_predicted_classes==valid_Y)[0]
print("%d correct labels" % len(correct))
for i, corr in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(valid_X[corr].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Label {}".format(valid_predicted_classes[corr], valid_Y[corr]))
    plt.tight_layout()

incorrect = np.where(valid_predicted_classes!=valid_Y)[0]
print("%d incorrect labels" % len(incorrect))
for i, incorr in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(valid_X[incorr].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted: {}, Label: {}".format(valid_predicted_classes[incorr], valid_Y[incorr]))
    plt.tight_layout()

## Classification Report

labels =  np.unique(train_Y)
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
label_map = dict(zip(labels, class_labels))

print(classification_report(valid_Y, valid_predicted_classes, target_names=class_labels))

## Confusion Matrices

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

validation_matrices = multilabel_confusion_matrix(valid_Y, valid_predicted_classes)
for i in range(10):
  print(class_labels[i],"\n",validation_matrices[i],"\n")

validation_matrix = confusion_matrix(valid_Y, valid_predicted_classes)
print(validation_matrix)

valid_disp = ConfusionMatrixDisplay(confusion_matrix = validation_matrix, display_labels=class_labels)
valid_fig, valid_ax = plt.subplots(figsize=(10,10))
valid_disp.plot(cmap=plt.cm.Blues,ax=valid_ax)

# Evaluation (Test data)

test_predicted_classes = best_model.predict(test_X)
test_predicted_classes = np.argmax(np.round(test_predicted_classes),axis=1)

test_eval = best_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

correct = np.where(test_predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, corr in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[corr].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Label {}".format(test_predicted_classes[corr], test_Y[corr]))
    plt.tight_layout()

incorrect = np.where(test_predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorr in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorr].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted: {}, Label: {}".format(test_predicted_classes[incorr], test_Y[incorr]))
    plt.tight_layout()

## Classification Report

print(classification_report(test_Y, test_predicted_classes, target_names=class_labels))

## Confusion Matrices

test_matrices = multilabel_confusion_matrix(test_Y, test_predicted_classes)
for i in range(10):
  print(class_labels[i],"\n",test_matrices[i],"\n")

test_matrix = confusion_matrix(test_Y, test_predicted_classes)
print(test_matrix)

test_disp = ConfusionMatrixDisplay(confusion_matrix = test_matrix, display_labels=class_labels)
test_fig, test_ax = plt.subplots(figsize=(10,10))
test_disp.plot(cmap=plt.cm.Blues,ax=test_ax)

## Evaluation on shirts

actual_shirts_index = np.where(test_Y == 6)[0]
test_X_shirts = test_X[actual_shirts_index]
actual_shirts = test_Y[actual_shirts_index]
prediction_for_actual_shirts = test_predicted_classes[actual_shirts_index]
incorrect_shirts = np.where(test_Y[actual_shirts_index] != test_predicted_classes[actual_shirts_index])[0]
for i, incorr in enumerate(incorrect_shirts[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X_shirts[incorr].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted: {}, Label: {}".format(prediction_for_actual_shirts[incorr], actual_shirts[incorr]))
    plt.tight_layout()
