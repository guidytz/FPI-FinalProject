import math
import numpy as np

def RF(img, der, sigma):
    a = math.exp(-math.sqrt(2) / sigma)
    V = a ** der
    F = img
    [lar, can] = np.array(img).shape[1:]

    for i in range(1, lar):
        for c in range(can):
            F[:, i, c] = F[:, i, c] + V[:, i] * (F[:, i - 1, c] - F[:, i, c])

    for i in range(lar-2, -1, -1):
        for c in range(can):
            F[:, i, c] = F[:, i, c] + V[:, i+1] * (F[:, i + 1, c] - F[:, i, c])

    return F

def NC(img, sigma_s, sigma_r, n_it=3, joint_img=None):
    pass

def IC(img, sigma_s, sigma_r, n_it=3, joint_img=None):
    pass