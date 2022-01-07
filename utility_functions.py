import cv2
import numpy as np
from opencv_functions_class import OpencvFunctions
import pytesseract


class UtilityFunctions(OpencvFunctions):
    def show(self):
        """Вывод изображения"""
        cv2.imshow('0', self.image)
        cv2.waitKey(0)


if __name__ == "__main__":
    img = UtilityFunctions("res/csv7.jpg")
    img.show()
    img.normalize()
    img.show()
    img.get_coords()
    img.show_cells()
