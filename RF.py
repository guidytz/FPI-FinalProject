import numpy as np
import cv2
import functions as fn

img = cv2.imread('statue_very_small.png')

[alt,lar,can]  = np.array(img).shape

print(alt,lar,can)

img_norm = img/255

dIdx = np.zeros((alt,lar))
dIdy = np.zeros((alt,lar))

dIcdx = np.diff(img_norm, 1, 1)
dIcdy = np.diff(img_norm, 1, 0)

for i in range(can):
    dIdx[:,1:] = dIdx[:,1:] + abs(dIcdx[:,:,i])
    dIdy[1:,:] = dIdy[1:,:] + abs(dIcdy[:,:,i])

print(dIdy)


#fn.printMat(dIcdy)



