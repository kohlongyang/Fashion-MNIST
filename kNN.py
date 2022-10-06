import pandas as pd
from sklearn import neighbors

def init():
    train = pd.read_csv('Dataset/fashion-mnist_train.csv')
    test = pd.read_csv('Dataset/fashion-mnist_test.csv')
    x_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    x_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values

    return x_train, y_train, x_test, y_test

def init_augmented():
    train_augmented = pd.read_csv('Dataset/augmented-fashion-mnist_train.csv')
    x_train_augmented = train_augmented.iloc[:, 1:].values
    y_train_augmented = train_augmented.iloc[:, 0].values

    return x_train_augmented, y_train_augmented

def knn(k, x_train, y_train, x_test, y_test):
    knn_init = neighbors.KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_model = knn_init.fit(x_train, y_train)
    return knn_model.score(x_test, y_test)

def run():
    x_train, y_train, x_test, y_test = init()
    k_values = [1, 3, 5]
    x_train_augmented, y_train_augmented = init_augmented()

    for i in range(len(k_values)):
        k = k_values[i]
        accuracy = knn(k, x_train, y_train, x_test, y_test)
        accuracy_augmented = knn(k, x_train_augmented, y_train_augmented, x_test, y_test)

        print('kNN accuracy for k = {k}: {accuracy}'.format(k=k, accuracy=accuracy))
        print('augmented kNN accuracy for k = {k}: {accuracy}'.format(k=k, accuracy=accuracy_augmented))

if __name__ == "__main__":
   run()