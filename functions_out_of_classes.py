import cv2
import numpy as np
from tensorflow import keras
import itertools


def custom_tuple_sorting(s, t, offset=4):
    x0, y0, _, _ = s
    x1, y1, _, _ = t
    if abs(y0 - y1) > offset:
        if y0 < y1:
            return -1
        else:
            return 1
    else:
        if x0 < x1:
            return -1

        elif x0 == x1:
            return 0

        else:
            return 1


def find_cell_contours(frame_image, crop_image):
    white_pixels = np.where(frame_image == 255)
    y = white_pixels[0]
    x = white_pixels[1]
    for i in range(len(y)):
        crop_image[y[i]][x[i]] = 255
    return crop_image


def find_digit_coordinates(image):
    cnts, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sort_contours(cnts, method="left-to-right")[1:]
    all_contours = []
    for i in range(0, len(bounding_boxes)):
        x, y, w, h = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], bounding_boxes[i][3]
        if h > 20 and w > 10:
            digit_coordinates = [x, y, x + w, y + h]
            all_contours.append(digit_coordinates)
    return all_contours


def detect_contour_in_contours(all_contours):
    for rec1, rec2 in itertools.permutations(all_contours, 2):
        if rec2[0] >= rec1[0] and rec2[1] >= rec1[1] and rec2[2] <= rec1[2] and rec2[3] <= rec1[3]:
            in_rec = [rec2[0], rec2[1], rec2[2], rec2[3]]
            all_contours.remove(in_rec)
    return all_contours


def image_binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return img_bin


def crop_digit(image, x0, y0, x1, y1):
    img_crop = image[y0 - 2:y1 + 2, x0 - 2:x1 + 2]
    res_crop_img = cv2.resize(img_crop, (28, 28))
    prediction_digit = predicting(res_crop_img)
    return prediction_digit


def predicting(image):
    img = keras.preprocessing.image
    model = keras.models.load_model('network/models/model.h5')
    x = img.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    return np.argmax(classes[0])
