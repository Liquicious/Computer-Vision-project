from cv2 import cv2


def loading_image():
    image = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    loading_image()
