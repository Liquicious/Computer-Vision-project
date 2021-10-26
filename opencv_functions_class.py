import cv2
import numpy as np


list_of_cells = {'Head': False, 'Q1': False, 'Ans1': True, 'Q2': False, 'Ans2': True, 'Q3': False,'Ans3': True,
                 'Q4': False, 'Ans4': True, 'Q5': False, 'Ans5': True, 'Q6': False, 'Ans6': True,
                 'Q7': False, 'Ans7': True}


class OpencvImage:
    def __init__(self, image_file):
        self.image = image_file
        self.border = None
        self.cells = list_of_cells

    def show(self):
        cv2.imshow('0', self.image)
        cv2.waitKey(0)

    def load(self):
        self.image = cv2.imread(self.image)

    def normalize(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        coords = np.column_stack(np.where(img_bin == 255))
        angle = 90 - cv2.minAreaRect(coords)[-1]
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        self.image = rotated

    def extract_lines(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        structuring_element = np.ones((1, 50), np.uint8)
        erode_image = cv2.erode(img_bin, structuring_element, iterations=1)
        hor_dilate_image = cv2.dilate(erode_image, structuring_element, iterations=1)

        structuring_element = np.ones((50, 1), np.uint8)
        erode_image = cv2.erode(img_bin, structuring_element, iterations=1)
        ver_dilate_image = cv2.dilate(erode_image, structuring_element, iterations=1)
        return hor_dilate_image,  ver_dilate_image

    def merge_lines(self, horizontal_lines, vertical_lines):
        structuring_element = np.ones((3, 3), np.uint8)
        merge_image = horizontal_lines + vertical_lines
        merge_image = cv2.dilate(merge_image, structuring_element, iterations=2)
        self.border = merge_image


img = OpencvImage("res/csv.jpg")

img.load()
img.show()

img.normalize()
img.show()

hor, ver = img.extract_lines()
img.merge_lines(hor, ver)

cv2.imshow('0', img.border)
cv2.waitKey(0)
