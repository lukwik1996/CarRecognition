import matplotlib
matplotlib.use("Agg")

import os
from keras.models import model_from_json, Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import LeakyReLU, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from skimage import io, transform
from sklearn.preprocessing import LabelBinarizer
import argparse
import cv2
import matplotlib.pyplot as plt


# CNN model structure
def cnn(classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D((2, 2), padding='same'))
    Dropout(0.25)

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    Dropout(0.25)

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    Dropout(0.5)

    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


# Generates images similar to the ones in dataset so there is more training/testing data
def create_dataset(dirTrain, dirTest):

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.0,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for folder in os.listdir(dirTrain):

        if len([name for name in os.listdir(dirTrain+folder) if os.path.isfile(os.path.join(dirTrain+folder, name))]) >= 15:
            max_i = 25
        else:
            max_i = 40

        for image in os.listdir(dirTrain + folder + "\\"):

            img = dirTrain + folder + "\\" + image
            img = load_img(img)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0

            for _ in datagen.flow(x, batch_size=1, save_to_dir=(dirTrain + folder + "\\"), save_prefix='gen',
                                  save_format='jpg'):
                i += 1
                if i > max_i:
                    break

    for folder in os.listdir(dirTest):

        if len([name for name in os.listdir(dirTest+folder) if os.path.isfile(os.path.join(dirTest+folder, name))]) >= 5:
            max_i = 25
        else:
            max_i = 40

        for image in os.listdir(dirTest + folder + "\\"):

            img = dirTest + folder + "\\" + image
            img = load_img(img)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0

            for _ in datagen.flow(x, batch_size=1, save_to_dir=(dirTest + folder + "\\"), save_prefix='gen',
                                  save_format='jpg'):
                i += 1
                if i > max_i:
                    break


# Reads the image, normalizes it and scales it down to input it into cnn
def image_transform(image):

    img = io.imread(image)
    img = img / 255.0
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')

    return img


# Function for predicting images from test folder (saves images with top 5 labels)
def predict(data_test, labels):
    i = 0
    index = 0
    imgs = []
    imgnames = []
    value = [0, 0, 0]

    for folder in os.listdir(dirTest):
        img_path = dirTest + folder + "\\"
        index += 1

        for image in os.listdir(img_path):
            img = cv2.imread(filename=img_path+image, flags=cv2.IMREAD_UNCHANGED)
            imgs.append(img)
            imgnames.append(os.path.basename(folder))

    for image, img in zip(data_test, imgs):

        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        predictions = predictions[0]
        top5 = np.array(predictions)
        top = top5.argsort()[-5:][::-1]

        img = cv2.copyMakeBorder(img, top=80, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=value)
        i += 1
        cv2.putText(img, labels[top[0]] + ": " + str(predictions[top[0]]),
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[1]] + ": " + str(predictions[top[1]]),
                    org=(5, 30),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[2]] + ": " + str(predictions[top[2]]),
                    org=(5, 45),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[3]] + ": " + str(predictions[top[3]]),
                    org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[4]] + ": " + str(predictions[top[4]]),
                    org=(5, 75),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        if not os.path.exists(prediction + imgnames[i-1]):
            os.mkdir(prediction + imgnames[i-1])
        img_name = prediction + imgnames[i-1] + "\\" + str(i) + ".jpg"

        cv2.imwrite(img_name, img)


# Loads the dataset into arrays
def load_data():
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    labels = []
    classes = 0

    for folder in os.listdir(dirTrain):
        img_path = dirTrain + folder + "\\"
        labels.append(os.path.basename(folder))
        classes += 1

        for image in os.listdir(img_path):
            img = image_transform(img_path + image)
            train_labels.append(classes - 1)
            train_data.append(img)

    index = 0
    for folder in os.listdir(dirTest):
        img_path = dirTest + folder + "\\"
        index += 1

        for image in os.listdir(img_path):
            img = image_transform(img_path + image)
            test_labels.append(index - 1)
            test_data.append(img)

    return np.array(train_data), np.array(test_data), np.array(train_labels), np.array(test_labels), classes, labels


if __name__ == "__main__":

    epochs_n = 100
    img_size = 50
    input_shape = (img_size, img_size, 3)

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save", help='path to save model')
    ap.add_argument("-l", "--load", help='path to load model')
    ap.add_argument("-p", "--plot", help='path to output accuracy/loss plot')
    ap.add_argument("-d", "--dataset", help='path to dataset')
    args = vars(ap.parse_args())

    prediction = "predictions\\"
	dirTrain = args['dataset'] + "\\train\\"
	dirTest = args['dataset'] + "\\test\\"
	
    # create_dataset(dirTrain, dirTest)
    data_train, data_test, labels_train, labels_test, num_classes, labels = load_data()

    encoder = LabelBinarizer()
    transformed_labels_train = encoder.fit_transform(labels_train)
    transformed_labels_test = encoder.fit_transform(labels_test)

    if args['save']:
        # Save model
        model = cnn(num_classes)

        H = model.fit(x=data_train,
                      y=transformed_labels_train,
                      epochs=epochs_n,
                      batch_size=128,
                      verbose=1,
                      validation_data=(data_test, transformed_labels_test))

        model_json = model.to_json()
        with open(args['save'] + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(args['save'] + ".h5")
        print("[INFO] model saved")

        (eval_loss, eval_accuracy, eval_top5_acc) = model.evaluate(x=data_test, y=transformed_labels_test, verbose=1)
        print("[INFO] accuracy: {: .2f}%".format(eval_accuracy * 100))
        print("[INFO] top5 accuracy: {: .2f}%".format(eval_top5_acc * 100))

        # Wykresy loss i accuracy
        plt.style.use('ggplot')
        N = epochs_n

        plt.figure()
        plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
        plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
        plt.title('Training loss')
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend(loc='lower left')
        plt.savefig('plot_loss.png')

        plt.figure()
        plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
        plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
        plt.plot(np.arange(0, N), H.history['top_k_categorical_accuracy'], label='top5_train_acc')
        plt.plot(np.arange(0, N), H.history['val_top_k_categorical_accuracy'], label='top5_val_acc')
        plt.title('Training accuracy')
        plt.xlabel("Epoch")
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig('plot_acc.png')

    if args['load']:

        # Load model
        json_file = open(args['load'] + ".json", 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(args['load'] + ".h5")
        print("[INFO] model wczytany")

        (eval_loss, eval_accuracy, eval_top5_acc) = model.evaluate(x=data_test, y=transformed_labels_test, verbose=1)
        print("[INFO] accuracy: {: .2f}%".format(eval_accuracy * 100))
        print("[INFO] top5 accuracy: {: .2f}%".format(eval_top5_acc * 100))

    file_path = 'labels.txt'
    file = open(file_path, 'w')

    for label in labels:
        file.write(label + '\n')
    file.close()

    # predict(data_test, labels)