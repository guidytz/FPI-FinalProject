import math, sys
import cv2
import numpy as np
import functions as fn
import filters as ft

img = cv2.imread('statue.png')

sigS = 200
sigR = 0.3
it = 3
ep_filter = 'RF'


# Normalize the image
img_norm = img/255

# Get the parameters for the filter using Domain Transform
if ep_filter == 'RF':
    [horDer, verDer] = fn.DT(img_norm,sigS,sigR,False)
else:
    [horInt, verInt] = fn.DT(img_norm,sigS,sigR,True)

# Initialize the H sigma to be used next
sigH = sigS

# Initialize the output image
img_out = img_norm

# Aplly the choosen filter 
for i in range(it-1):
    # Calculate the current sigma H using equation 14 of the paper
    curSigH = sigH * math.sqrt(3) * (2**(it-(i+1))) / math.sqrt(4**it - 1)
    
    # If the filter is not RF, we'll need the filter_radius
    if ep_filter != 'RF':
        filter_redius = math.sqrt(3) * curSigH
    
    # Apply the filter 
    if ep_filter == 'RF':
        img_out = ft.RF(img_out, horDer, curSigH)
    elif ep_filter == 'IC':
        img_out = None
    else:
        img_out = None

    # Transpose the imagem so we can apply the filter vertically
    img_out = fn.transposeImage(img_out)

    if ep_filter == 'RF':
        img_out = ft.RF(img_out, np.transpose(verDer), curSigH)
    elif ep_filter == 'IC':
        img_out = None
    else:
        img_out = None
    
    img_out = fn.transposeImage(img_out)


cv2.imshow('Imagem filtrada', img_out)

key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows()
