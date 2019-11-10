import numpy as np
import cv2

img = cv2.imread('statue_very_small.png')

alt = np.array(img).shape[0]
lar = np.array(img).shape[1]
can = np.array(img).shape[2]

img_norm = img/255

dIcdy = np.diff(img_norm, 1, 1)
dIcdy = np.diff(img_norm, 1, 0)

print(np.array(dIcdy).shape)


## print formatado semelhante ao MATLAB para comparação de resultados
for i in range(len(dIcdy)):
    for j in range(len(dIcdy[i])):
        print("%.4f" % dIcdy[i][j][0], end = ' ')
    print('')
print('')
for i in range(len(dIcdy)):
    for j in range(len(dIcdy[i])):
        print("%.4f" % dIcdy[i][j][1], end = ' ')
    print('')
print('')
for i in range(len(dIcdy)):
    for j in range(len(dIcdy[i])):
        print("%.4f" % dIcdy[i][j][2], end = ' ')
    print('')