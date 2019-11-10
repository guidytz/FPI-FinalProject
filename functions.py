import math
import numpy as np

# print formatado semelhante ao MATLAB para compara��o de resultados


def printMat(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print("%+.4f" % arr[i][j][0], end=' ')
        print('')
    print('')
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print("%+.4f" % arr[i][j][1], end=' ')
        print('')
    print('')
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print("%+.4f" % arr[i][j][2], end=' ')
        print('')


def rf(img, der, sigma):
    a = math.exp(-math.sqrt(2) / sigma)
    V = a ** der
    F = img
    [alt, lar, can] = np.array(img).shape

    for i in range(1, lar):
        for c in range(can):
            F[:, i, c] = F[:, i, c] + V[:, i] * (F[:, i - 1, c] - F[:, i, c])

    for i in range(lar-2, -1, -1):
        for c in range(can):
            F[:, i, c] = F[:, i, c] + V[:, i+1] * (F[:, i + 1, c] - F[:, i, c])

    return F


def transpose_image(img):
    [h, w, num_channels] = np.array(img).shape
    T = np.zeros((w, h, num_channels))
    for c in range(num_channels):
        T[:, :, c] = np.transpose(img[:, :, c])
    return T
