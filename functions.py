"""File containg auxiliar functions for applying the filters."""

# Imports
import numpy as np


def print_3d(arr):
    """Temporary function to print 3D arrays like in MATLAB. Remove before
    submission."""

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


def image_transpose(img):
    """Transposes an image in Numpy array
    Because we have 3 channels and we don't want to transpose then, this
    function preserves the third dimension of the image's Numpy array"""
    [alt, lar, can] = np.array(img).shape
    t_img = np.zeros((lar, alt, can))
    for i in range(can):
        t_img[:, :, i] = np.transpose(img[:, :, i])
    return t_img


def domain_transform(img, sigma_s, sigma_r, integrate):
    """Domain Transform function
    Returns either the parcial derivatives of the Domain transform equation
    (the equation 11 in the paper) if 'integrate' is False, or the actual
    integrated components if True."""

    [alt, lar, can] = np.array(img).shape

    # Calculate parcial derivatives of Ik of the equation
    pd_ik_x = np.diff(img, 1, 1)
    pd_ik_y = np.diff(img, 1, 0)

    # Initialize the summation matrices of the equation
    sum_pd_ik_x = np.zeros((alt, lar))
    sum_pd_ik_y = np.zeros((alt, lar))

    # Perform the summation
    for i in range(can):
        sum_pd_ik_x[:, 1:] = sum_pd_ik_x[:, 1:] + abs(pd_ik_x[:, :, i])
        sum_pd_ik_y[1:, :] = sum_pd_ik_y[1:, :] + abs(pd_ik_y[:, :, i])

    # Finally, calculate the rest of the equation inside the integral
    hor_differences = (1 + sigma_s/sigma_r * sum_pd_ik_x)
    ver_differences = (1 + sigma_s/sigma_r * sum_pd_ik_y)

    if integrate:
        # Perform the integration if required
        return (np.cumsum(hor_differences, 1),
                np.cumsum(ver_differences, 0))

    return (hor_differences, ver_differences)
