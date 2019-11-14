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
    [hgt, wid, chan] = np.array(img).shape
    t_img = np.zeros((wid, hgt, chan))
    for i in range(chan):
        t_img[:, :, i] = np.transpose(img[:, :, i])
    return t_img


def domain_transform(img, sigma_s, sigma_r, integrate):
    """Domain Transform function
    Returns either the parcial derivatives of the Domain transform equation
    (the equation 11 in the paper) if 'integrate' is False, or the actual
    integrated components if True."""

    [hgt, wid, chan] = np.array(img).shape

    # Calculate parcial derivatives of Ik of the equation
    pd_ik_x = np.diff(img, 1, 1)
    pd_ik_y = np.diff(img, 1, 0)

    # Initialize the summation matrices of the equation
    sum_pd_ik_x = np.zeros((hgt, wid))
    sum_pd_ik_y = np.zeros((hgt, wid))

    # Perform the summation
    for i in range(chan):
        sum_pd_ik_x[:, 1:] = sum_pd_ik_x[:, 1:] + abs(pd_ik_x[:, :, i])
        sum_pd_ik_y[1:, :] = sum_pd_ik_y[1:, :] + abs(pd_ik_y[:, :, i])

    # Apply the sigmas
    hor_differences = (1 + sigma_s/sigma_r * sum_pd_ik_x)
    ver_differences = (1 + sigma_s/sigma_r * sum_pd_ik_y)

    if integrate:
        # Perform the integration if required
        return (np.cumsum(hor_differences, 1),
                np.cumsum(ver_differences, 0))

    return (hor_differences, ver_differences)


def limits_indexes(transform, hgt, wdt, radius):
    """ This function returns the indexes of the pixels associated with the
    upper and lower limits of a box kernel"""

    # Get the lower and upper limits of the kernel
    lower_limits = transform - radius
    upper_limits = transform + radius

    #print(lower_limits)

    # Initialize the pixels matrices
    lower_indexes = np.zeros(np.array(transform).shape)
    upper_indexes = np.zeros(np.array(transform).shape)

    for row in range(hgt):
        transform_row = np.concatenate([transform[row, :], np.array([np.inf])])

        lower_limits_row = lower_limits[row, :]
        upper_limits_row = upper_limits[row, :]

        lower_indexes_row = np.zeros([wdt])
        upper_indexes_row = np.zeros([wdt])

        lower_indexes_row[0] = (
            transform_row > lower_limits_row[0]).nonzero()[0][0]
        upper_indexes_row[0] = (
            transform_row > upper_limits_row[0]).nonzero()[0][0]

        for col in range(1, wdt):
            lower_indexes_row[col] = lower_indexes_row[col-1] + (
                transform_row[int(lower_indexes_row[col-1]):] > lower_limits_row[col]).nonzero()[0][0]
            upper_indexes_row[col] = upper_indexes_row[col-1] + (
                transform_row[int(upper_indexes_row[col-1]):] > upper_limits_row[col]).nonzero()[0][0]

        lower_indexes[row, :] = lower_indexes_row + 1
        upper_indexes[row, :] = upper_indexes_row + 1

    return (lower_indexes, upper_indexes)
