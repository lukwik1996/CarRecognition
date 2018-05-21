import numpy as np
from skimage import io, transform
from keras.models import model_from_json
import cv2
import argparse
import os


def image_transform(image):

    img = io.imread(image)
    img = img / 255.0
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')

    return img


def predict_image(image, labels):

    value = [0, 0, 0]

    im = image_transform(image)
    image = cv2.imread(image)

    im = np.expand_dims(im, axis=0)
    predictions = model.predict(im)
    predictions = predictions[0]
    top5 = np.array(predictions)
    top = top5.argsort()[-5:][::-1]

    top1 = labels[top[0]].split('_')
    top2 = labels[top[1]].split('_')
    top3 = labels[top[2]].split('_')
    top4 = labels[top[3]].split('_')
    top5 = labels[top[4]].split('_')

    text_top1 = top1[0].capitalize() + ' ' + top1[1].capitalize()
    if len(top1) > 2:
        text_top1 = text_top1 + ' ' + top1[2].capitalize()

    text_top2 = top2[0].capitalize() + ' ' + top2[1].capitalize()
    if len(top2) > 2:
        text_top2 = text_top2 + ' ' + top2[2].capitalize()

    text_top3 = top3[0].capitalize() + ' ' + top3[1].capitalize()
    if len(top3) > 2:
        text_top3 = text_top3 + ' ' + top3[2].capitalize()

    text_top4 = top4[0].capitalize() + ' ' + top4[1].capitalize()
    if len(top4) > 2:
        text_top4 = text_top4 + ' ' + top4[2].capitalize()

    text_top5 = top5[0].capitalize() + ' ' + top5[1].capitalize()
    if len(top5) > 2:
        text_top5 = text_top5 + ' ' + top5[2].capitalize()

    img = cv2.copyMakeBorder(image, top=80, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=value)

    cv2.putText(img, text_top1 + '  ' + str(predictions[top[0]]),
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                lineType=2)

    cv2.putText(img, text_top2 + '  ' + str(predictions[top[1]]),
                org=(5, 30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                lineType=2)

    cv2.putText(img, text_top3 + '  ' + str(predictions[top[2]]),
                org=(5, 45),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                lineType=2)

    cv2.putText(img, text_top4 + '  ' + str(predictions[top[3]]),
                org=(5, 60),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                lineType=2)

    cv2.putText(img, text_top5 + '  ' + str(predictions[top[4]]),
                org=(5, 75),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                lineType=2)

    cv2.imwrite('image', img)
    cv2.waitKey(0)


def predict_folder(folder, labels):

    value = [0, 0, 0]

    for image in os.listdir(folder):

        image = folder + "\\" + image
        im = image_transform(image)
        image = cv2.imread(image)

        im = np.expand_dims(im, axis=0)
        predictions = model.predict(im)
        predictions = predictions[0]
        top5 = np.array(predictions)
        top = top5.argsort()[-5:][::-1]

        top1 = labels[top[0]].split('_')
        top2 = labels[top[1]].split('_')
        top3 = labels[top[2]].split('_')
        top4 = labels[top[3]].split('_')
        top5 = labels[top[4]].split('_')

        text_top1 = top1[0].capitalize() + ' ' + top1[1].capitalize()
        if len(top1) > 2:
            text_top1 = text_top1 + ' ' + top1[2].capitalize()

        text_top2 = top2[0].capitalize() + ' ' + top2[1].capitalize()
        if len(top2) > 2:
            text_top2 = text_top2 + ' ' + top2[2].capitalize()

        text_top3 = top3[0].capitalize() + ' ' + top3[1].capitalize()
        if len(top3) > 2:
            text_top3 = text_top3 + ' ' + top3[2].capitalize()

        text_top4 = top4[0].capitalize() + ' ' + top4[1].capitalize()
        if len(top4) > 2:
            text_top4 = text_top4 + ' ' + top4[2].capitalize()

        text_top5 = top5[0].capitalize() + ' ' + top5[1].capitalize()
        if len(top5) > 2:
            text_top5 = text_top5 + ' ' + top5[2].capitalize()
        img = cv2.copyMakeBorder(image, top=80, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=value)

        cv2.putText(img, text_top1,
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    lineType=2)

        cv2.putText(img, text_top2,
                    org=(5, 30),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, text_top3,
                    org=(5, 45),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, text_top4,
                    org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, text_top5,
                    org=(5, 75),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == '__main__':

    img_size = 50

    input_shape = (img_size, img_size, 3)

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", help='path to load model')
    ap.add_argument("-n", "--names", help='path to labels file')
    ap.add_argument("-i", "--image", help='path to input image')
    args = vars(ap.parse_args())

    if args['model'] is None:
        args['model'] = 'MakeModelRecognition'

    if args['names'] is None:
        args['names'] = 'labels.txt'

    # Load model
    json_file = open(args['model'] + ".json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(args['model'] + ".h5")
    print("[INFO] model loaded")

    image_path = args['image']

    labels_path = args['names']
    file = open(labels_path, 'r')
    labels_array = []

    for label in file:
        labels_array.append(label[:-1])
    file.close()

    if os.path.isfile(image_path):
        predict_image(image_path, labels_array)
    else:
        predict_folder(image_path, labels_array)
