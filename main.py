import cv2
import numpy as np


def show(img):
    cv2.imshow('0', img)
    cv2.waitKey(0)


def normalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(img_bin == 255))
    angle = 90 - cv2.minAreaRect(coords)[-1]
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def extract_lines(img, type):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    if type == 'h':
        structuring_element = np.ones((1, 50), np.uint8)
    elif type == 'v':
        structuring_element = np.ones((50, 1), np.uint8)

    erode_image = cv2.erode(img_bin, structuring_element, iterations=1)
    dilate_image = cv2.dilate(erode_image, structuring_element, iterations=1)
    return dilate_image


def merge_lines(horizontal_lines, vertical_lines):
    structuring_element = np.ones((3, 3), np.uint8)
    merge_image = horizontal_lines + vertical_lines
    merge_image = cv2.dilate(merge_image, structuring_element, iterations=2)
    return merge_image


img = cv2.imread("res/csv.jpg")
rotated = normalize(img)

show(rotated)
hor = extract_lines(rotated, "h")
ver = extract_lines(rotated, "v")
show(merge_lines(hor, ver))
