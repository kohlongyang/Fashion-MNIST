import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import cross_validate

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

def find_best_k():
    x_train, y_train, x_test, y_test = init()
    k_values = [1, 3, 5]
    best_k = -1
    best_validation_score = -1
    k_cv_score = {}

    for i in range(len(k_values)):
        k = k_values[i]

        knn_model = neighbors.KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        cv = cross_validate(knn_model, x_train, y_train, cv=3)
        average_cv = cv['test_score'].mean()
        k_cv_score[k] = average_cv
        if average_cv >= best_validation_score:
            best_validation_score = average_cv
            best_k = k

    print(k_cv_score)
    print('best k = {k}: {best_validation_score}'.format(k=best_k, best_validation_score=best_validation_score))
    return best_k

def test(k):
    x_train, y_train, x_test, y_test = init()
    x_train_augmented, y_train_augmented = init_augmented()

    accuracy = knn(k, x_train, y_train, x_test, y_test)
    accuracy_augmented = knn(k, x_train_augmented, y_train_augmented, x_test, y_test)

    print('kNN accuracy for k = {k}: {accuracy}'.format(k=k, accuracy=accuracy))
    print('augmented kNN accuracy for k = {k}: {accuracy}'.format(k=k, accuracy=accuracy_augmented))

def run():
    k = find_best_k()
    test(k)

if __name__ == "__main__":
    run()