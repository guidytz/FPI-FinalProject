"""File containing the main filter functions"""

# Imports
import math
import numpy as np

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
            img[:, i, j] = img[:, i, j] + a_pow_d[:, i] * (img[:, i - 1, j] - img[:, i, j])

    # Apply the filter from right to left, achieving symmetric response
    for i in range(wid-2, -1, -1):
        for j in range(chan):
            img[:, i, j] = img[:, i, j] + a_pow_d[:, i+1] * (img[:, i + 1, j] - img[:, i, j])

    return img

def normalized_convolution(img, sigma_s, sigma_r, n_it=3, joint_img=None):
    pass

def interpolated_convolution(img, sigma_s, sigma_r, n_it=3, joint_img=None):
    pass