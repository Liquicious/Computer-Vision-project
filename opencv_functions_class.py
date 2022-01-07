import cv2
import numpy as np
import itertools
import pytesseract


class OpencvFunctions:
    def __init__(self, image_file):
        self.dots = None
        self.rects_coords = None
        self.image = cv2.imread(image_file)
        self.height, self.width = self.image.shape[:2]
        self.borders = None
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.image_bin = cv2.threshold(self.gray, 128, 255, cv2.THRESH_BINARY_INV)

    def load(self):
        """Загружает изображение"""
        self.image = cv2.imread(self.image)

    def normalize(self):
        """Приводит изображение в удобный для работы вид (выравнивание по горизонтали)"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        cords = np.column_stack(np.where(img_bin == 255))
        angle = cv2.minAreaRect(cords)[-1]

        if angle < 45:
            angle = -angle
        else:
            angle = 90 - angle

        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        self.image = rotated

    def get_coords(self):
        """Определение координат ячеек"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        # Выделяем горизонтальные линии при помощи ядра 1x50(горизонтальная линия)
        structuring_element = np.ones((1, 50), np.uint8)
        erode_image = cv2.erode(img_bin, structuring_element, iterations=1)
        hor = cv2.dilate(erode_image, structuring_element, iterations=1)
        # Выделяем вертикальные линии при помощи ядра 50x1(вертикальная линия)
        structuring_element = np.ones((50, 1), np.uint8)
        erode_image = cv2.erode(img_bin, structuring_element, iterations=1)
        ver = cv2.dilate(erode_image, structuring_element, iterations=1)
        # Применяем бинарное умножение  вертикальных и горизонтальных линий для получения точек пересечения
        self.dots = cv2.bitwise_and(ver, hor)

        coords = []

        # Определяем координаты точек пересечения
        for y in range(self.height):
            for x in range(self.width):
                if self.dots[y][x] == 255:
                    coords.append((x, y))

        coords.pop(-1)

        self.rects_coords = []

        # Для каждой ячейки берем правый верхний и левый нижний угол по координатам
        for i in range(0, len(coords) - 1, 2):
            self.rects_coords.append((coords[i], coords[i + 1]))

    def show_cells(self):
        """Вывод каждой ячейки"""
        for x in self.rects_coords:
            if x[1][0] < x[0][0]:
                crop_img = self.image[x[0][1]:x[1][1], x[1][0]:x[0][0]]
            else:
                crop_img = self.image[x[0][1]:x[1][1], x[0][0]:x[1][0]]

            crop_img = cv2.threshold(crop_img, 170, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = np.ones((2, 2), np.uint8)
            crop_img = cv2.dilate(crop_img, kernel, iterations=2)
            crop_img = cv2.bitwise_not(crop_img)

            cv2.imshow('0', crop_img)
            cv2.waitKey(0)

            custom_config = r'--oem 3 --psm 6 outputbase digits'
            result = pytesseract.image_to_string(crop_img, config=custom_config)
            if result:
                print(result[0])
            else:
                print("Not identified")
