import cv2
import numpy as np
from opencv_functions_class import OpencvFunctions


class UtilityFunctions(OpencvFunctions):
    def show(self):
        cv2.imshow('0', self.image)
        cv2.waitKey(0)

    def show_borders(self):
        if self.borders is None:
            print("borders weren't created yet")
        else:
            cv2.imshow('0', self.borders)
            cv2.waitKey(0)


if __name__ == "__main__":
    img = UtilityFunctions("res/csv.jpg")
    img.load()
    img.show()
    img.normalize()
    img.show()
    hor, ver = img.extract_lines()
    img.merge_lines(hor, ver)
    img.show_borders()
