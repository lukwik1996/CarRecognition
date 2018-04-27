import tensorflow as tf
import os
import keras, keras.models, keras.layers
from skimage import io
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D, BatchNormalization
import numpy as np

dirTest = "D:\\si\\dataset\\test\\"
dirTrain = "D:\\si\\dataset\\train\\"
EPOCHS = 10


def build_cnn():
    model = keras.models.Sequential()
    model.add(Conv2D(96, (3, 3), strides = (1, 1), activation='relu', input_shape=(227, 227, 3)))    # Conv1
    model.add(MaxPooling2D((2, 2)))                                                                # Pool1
    model.add(BatchNormalization())                                                                # LRN1
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                               # Conv2
    model.add(MaxPooling2D((2, 2)))                                                                # Pool2
    model.add(BatchNormalization())                                                                # LRN2
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))                               # Conv3
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))                             # Conv4
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))                             # Conv5
    model.add(MaxPooling2D((2, 2)))                                                                # Pool3
    model.add(BatchNormalization())                                                                # LRN3


def load_data():

    train_data = []
    test_data = []
    train_labels = []

    for folder in os.listdir(dirTrain):
        train_labels.append(os.path.basename(folder))
        img = dirTrain + folder + "\\"

        for image in os.listdir(img):
            train_data.append(img + image)

    for folder in os.listdir(dirTest):
        img = dirTest + folder + "\\"

        for image in os.listdir(img):
            test_data.append(img + image)

    return np.array(train_data), np.array(test_data), np.array(train_labels)


data_train, data_test, labels = load_data()
