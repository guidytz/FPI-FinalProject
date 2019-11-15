"""File containing the main filter functions"""

# Imports
import math
import numpy as np
import functions as fn


def recursive_filtering(img, diff, sigma):
    """Recursive Filtering"""

    # Calculate the feedback coefficient with the equation explained in the
    # papers' appendix
    feedback_coef = math.exp(-math.sqrt(2) / sigma)

    # Calculate the a^d terms of the equation 21
    a_pow_d = feedback_coef ** diff
    [wid, chan] = np.array(img).shape[1:]

    # Apply the filter from left to right
    for i in range(1, wid):
        for j in range(chan):
            img[:, i, j] = img[:, i, j] + a_pow_d[:, i] * \
                (img[:, i - 1, j] - img[:, i, j])

    # Apply the filter from right to left, achieving symmetric response
    for i in range(wid-2, -1, -1):
        for j in range(chan):
            img[:, i, j] = img[:, i, j] + a_pow_d[:, i+1] * \
                (img[:, i + 1, j] - img[:, i, j])

    return img


def normalized_convolution(img, transform, box_radius):
    [hgt, wdt, chan] = np.array(img).shape
    [lower_indexes, upper_indexes] = fn.limits_indexes(
        transform, hgt, wdt, box_radius)
    

    # create a 'SAT' with the same size of the image
    sum_a_table = np.zeros((hgt, wdt+1, chan)) 
    
    # compute sum of all columns along each row in each channel
    sum_a_table[:,1:,:] = np.cumsum(np.array(img), 1) 

    upper_indexes = upper_indexes.astype(int)
    lower_indexes = lower_indexes.astype(int)

    for c in range(chan):
        for i in range(hgt):
            for j in range(wdt):
                img[i, j, c] = (sum_a_table[i, lower_indexes[i, j], c] \
                     - sum_a_table[i, upper_indexes[i, j], c]) / (upper_indexes[i, j] - lower_indexes[i, j])
    return img


def interpolated_convolution(img, transform, box_radius):
    [hgt, wdt, chan] = np.array(img).shape
    [lower_indexes, upper_indexes] = fn.limits_indexes(
        transform, hgt, wdt, box_radius)


    return img
