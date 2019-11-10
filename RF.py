import numpy as np
import cv2
import functions as fn
import math

img = cv2.imread('statue_very_small.png')

sigs = 200
sigr = 2
it = 3

[alt, lar, can] = np.array(img).shape

print(alt, lar, can)

img_norm = img/255

didx = np.zeros((alt, lar))
didy = np.zeros((alt, lar))

dicdx = np.diff(img_norm, 1, 1)
dIcdy = np.diff(img_norm, 1, 0)

for i in range(can):
    didx[:, 1:] = didx[:, 1:] + abs(dicdx[:, :, i])
    didy[1:, :] = didy[1:, :] + abs(dIcdy[:, :, i])

dhdx = (1 + sigs/sigr * didx)
dvdy = np.transpose(1 + sigs/sigr * didy)

sigh = sigs

F = img_norm

for i in range(it-1):
    sigh_i = sigh * math.sqrt(3) * (2**(it-(i+1))) / math.sqrt(4**it - 1)
    F = fn.rf(F,dhdx,sigh_i)
    F = fn.transpose_image(F)
    F = fn.rf(F,dvdy,sigh_i)
    F = fn.transpose_image(F)


F = F*255

# falta converter o F para NumPy array uint8