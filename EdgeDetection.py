import cv2
import pandas as pd
import numpy as np
import csv


def process_training_data():
    processed_data = []
    count = 0
    with open('Dataset/fashion-mnist_train.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        processed_data = next(reader, None)
        for data in reader:
            label = data[0]
            pixels = data[1:]
            pixels = np.array(pixels, dtype='uint8')
            pixels = pixels.reshape((28, 28))

            img_blur = cv2.GaussianBlur(pixels, (3, 3), 0)
            edges = cv2.Canny(image=img_blur, threshold1=85, threshold2=255)  # Canny Edge Detection
            edges = edges.reshape((784,))
            row = np.append(label, edges)
            processed_data = np.vstack((processed_data, row))
            count += 1
            print(count)

    df = pd.DataFrame(processed_data)
    df.to_csv("Dataset/edge-processed-fashion-mnist_train.csv", index=False, header=False)


def process_testing_data():
    processed_data = []
    count = 0
    with open('Dataset/fashion-mnist_test.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        processed_data = next(reader, None)
        for data in reader:
            label = data[0]
            pixels = data[1:]
            pixels = np.array(pixels, dtype='uint8')
            pixels = pixels.reshape((28, 28))

            img_blur = cv2.GaussianBlur(pixels, (3, 3), 0)
            edges = cv2.Canny(image=img_blur, threshold1=85, threshold2=255)  # Canny Edge Detection
            edges = edges.reshape((784,))
            row = np.append(label, edges)
            processed_data = np.vstack((processed_data, row))
            count += 1
            print(count)

    df = pd.DataFrame(processed_data)
    df.to_csv("Dataset/edge-processed-fashion-mnist_test.csv", index=False, header=False)


def run():
    process_training_data()
    process_testing_data()


if __name__ == "__main__":
    run()