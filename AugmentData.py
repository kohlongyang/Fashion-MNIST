import numpy as np
import pandas as pd
from scipy.ndimage import shift

def init():
    train = pd.read_csv('Dataset/fashion-mnist_train.csv')
    test = pd.read_csv('Dataset/fashion-mnist_test.csv')
    x_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    x_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    cols = train.columns

    return x_train, y_train, x_test, y_test, cols

def augment_data(x_train, y_train):
    # ---Shifted Sets---
    right = np.zeros(shape=(len(x_train), len(x_train[0])), dtype="int64")
    left = np.zeros(shape=(len(x_train), len(x_train[0])), dtype="int64")
    up = np.zeros(shape=(len(x_train), len(x_train[0])), dtype="int64")
    down = np.zeros(shape=(len(x_train), len(x_train[0])), dtype="int64")

    for i in range(len(x_train)):
        r = shift(x_train[i].reshape(28, 28), [0, 1], cval=0).reshape(1, 784)
        l = shift(x_train[i].reshape(28, 28), [0, -1], cval=0).reshape(1, 784)
        u = shift(x_train[i].reshape(28, 28), [-1, 0], cval=0).reshape(1, 784)
        d = shift(x_train[i].reshape(28, 28), [1, 0], cval=0).reshape(1, 784)

        # Add shifted images to the shifted sets
        left[i] = l
        right[i] = r
        up[i] = u
        down[i] = d
    # Append the new data:
    x_train_augmented = np.copy(x_train)  # original data
    x_train_augmented = np.append(x_train_augmented, left, axis=0)  # left-shifted
    x_train_augmented = np.append(x_train_augmented, right, axis=0)  # right-shifted
    x_train_augmented = np.append(x_train_augmented, up, axis=0)  # up-shifted
    x_train_augmented = np.append(x_train_augmented, down, axis=0)  # down-shifted

    y_train_augmented = np.copy(y_train)
    y_train_augmented = np.append(y_train_augmented, y_train, axis=0)
    y_train_augmented = np.append(y_train_augmented, y_train, axis=0)
    y_train_augmented = np.append(y_train_augmented, y_train, axis=0)
    y_train_augmented = np.append(y_train_augmented, y_train, axis=0)

    return x_train_augmented, y_train_augmented

def generate_augmented_data_csv(cols, x_train_augmented, y_train_augmented):
    reshaped_y = y_train_augmented.reshape(-1, 1)
    data = np.concatenate((reshaped_y, x_train_augmented), axis=1)
    df = pd.DataFrame(data, columns=cols)
    df.to_csv("Dataset/augmented-fashion-mnist_train.csv", index=False)

def run():
    x_train, y_train, x_test, y_test, cols = init()
    x_train_augmented, y_train_augmented = augment_data(x_train, y_train)
    # generate_augmented_data_csv(cols, x_train_augmented, y_train_augmented)

if __name__ == "__main__":
    run()