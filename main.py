import numpy as np
import cv2

img = cv2.imread('lena.jpg', 1)
img = img / 255;
cv2.imshow('image', img)
k = cv2.waitKey(0)
print(np.array(img).shape)


if k == 27:
    cv2.destroyAllWindows()
