import math
import numpy as np

# For printing 3D matrices formatted as in MATLAB
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

# A function to transpose an image
def transposeImage(img):
    [h, w, num_channels] = np.array(img).shape
    T = np.zeros((w, h, num_channels))
    for c in range(num_channels):
        T[:, :, c] = np.transpose(img[:, :, c])
    return T


# Domain Transform function
# Returns either the parcial derivatives of the Domain transform equation
# (the equation 11 in the paper) if 'integrate' is False, or the actual
# components if True.
def DT(img,sigS,sigR,integrate):
    ## Calculate the domain transform using equation 11 of the paper
    [alt, lar, can] = np.array(img).shape

    # Calculate parcial derivatives of Ik of the equation 
    pdIkx = np.diff(img, 1, 1)
    pdIky = np.diff(img, 1, 0)

    # Initialize the summation matrices of the equation 
    sum_pdIkx = np.zeros((alt, lar))
    sum_pdIky = np.zeros((alt, lar))

    # Perform the summation
    for i in range(can):
        sum_pdIkx[:, 1:] = sum_pdIkx[:, 1:] + abs(pdIkx[:, :, i])
        sum_pdIky[1:, :] = sum_pdIky[1:, :] + abs(pdIky[:, :, i])

    # Finally, calculate the rest of the equation inside the integral
    horDer = (1 + sigS/sigR * sum_pdIkx)
    verDer = (1 + sigS/sigR * sum_pdIky)

    if integrate:
        # Perform the integration
        horInt = np.cumsum(horDer, 1)
        verInt = np.cumsum(verDer, 0)
        return (horInt, verInt)
    else:
        return (horDer, verDer)