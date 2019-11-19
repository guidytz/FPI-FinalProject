"""File containing the main filter functions"""

# Imports
import math
import sys
import numpy as np
import functions as fn

ELAPSED_STEP = 5


def recursive_filtering(img, diff, sigma):
    """Recursive Filtering"""

    [wdt, chan] = np.array(img).shape[1:]

    # Calculate the feedback coefficient
    feedback_coef = math.exp(-math.sqrt(2) / sigma)

    # Calculate the a^d terms
    a_pow_d = feedback_coef ** diff

    # Apply the filter from left to right
    for i in range(1, wdt):
        for j in range(chan):
            img[:, i, j] = img[:, i, j] + a_pow_d[:, i] * \
                (img[:, i - 1, j] - img[:, i, j])

    # Apply the filter from right to left, achieving symmetric response
    for i in range(wdt-2, -1, -1):
        for j in range(chan):
            img[:, i, j] = img[:, i, j] + a_pow_d[:, i+1] * \
                (img[:, i + 1, j] - img[:, i, j])

    return img


def normalized_convolution(img, transform, sigma):
    """Normalized Convolution"""

    [hgt, wdt, chan] = np.array(img).shape

    # Calculate the kernel's box radius
    box_radius = math.sqrt(3) * sigma

    # Get the upper and lower limits indices of the box kernel
    [lower_indices, upper_indices] = fn.limits_indices(
        transform, hgt, wdt, box_radius)

    # Create a 'SAT' with the same size of the image
    sum_a_table = np.zeros((hgt, wdt+1, chan))

    # Compute sum of all columns along each row in each channel
    sum_a_table[:, 1:, :] = np.cumsum(np.array(img), 1)

    for cha in range(chan):
        for row in range(hgt):
            for col in range(wdt):
                # Get the current indices
                cur_lower_index = lower_indices[row, col]
                cur_upper_index = upper_indices[row, col]

                # Calculate the resulting pixel
                img[row, col, cha] = ((sum_a_table[row, cur_lower_index, cha]
                                       - sum_a_table[row, cur_upper_index, cha])
                                      / (cur_upper_index - cur_lower_index))
    return img


def interpolated_convolution(img, transform, sigma):
    """Interpolated Convolution"""

    [hgt, wdt, chan] = np.array(img).shape

    # Calculate the kernel's box radius
    box_radius = math.sqrt(3) * sigma

    # Get the upper and lower limits indices of the box kernel
    [lower_indices, upper_indices] = fn.limits_indices(
        transform, hgt, wdt, box_radius)

    lower_limits = transform - box_radius
    upper_limits = transform + box_radius

    # Compute the areas using the trapzoidal rule
    bases = img[:, 1:, :] + img[:, :-1, :]
    heights = transform[:, 1:] - transform[:, :-1]
    areas = .5*bases

    for i in range(chan):
        areas[:, :, i] *= heights

    # Create a 'SAT' with the same size of the image
    sum_a_table = np.zeros((hgt, wdt, chan))

    # Compute sum of all columns along each row in each channel of the
    # areas matrix
    sum_a_table[:, 1:, :] = np.cumsum(np.array(areas), 1)

    # Pad the matrices representing the image, SAT and transform so we can
    # perform the interpolation without going out of bounds. For the transform
    # and the image, we assume those new pixels to be equal to their adjacent
    # ones. For the SAT, we use zeros.
    img_padded = np.pad(img, ((0, 0), (1, 1), (0, 0)), 'edge')
    sum_a_table = np.pad(sum_a_table, ((0, 0), (1, 1), (0, 0)), 'constant')
    transform = np.pad(transform, ((0, 0), (1, 1)), 'edge')

    # Adjust the tranform's new borders
    transform[:, 0] = transform[:, 0] - 1.2*box_radius
    transform[:, -1] = transform[:, -1] + 1.2*box_radius

    # Create the indices matrices for the interpolations
    lower_indices_plus = lower_indices + 1
    upper_indices_plus = upper_indices + 1

    # Initialize the matrices
    center_areas = np.zeros(img.shape)
    left_aux = np.zeros(img.shape)
    left_areas = np.zeros(img.shape)
    right_aux = np.zeros(img.shape)
    right_areas = np.zeros(img.shape)

    alpha_l = np.zeros((hgt, wdt))
    alpha_r = np.zeros((hgt, wdt))

    # Apply the interpolation
    for row in range(hgt):
        for col in range(wdt):
            # Get the current indices to use in the equations
            cur_lower_index = lower_indices[row, col]
            cur_upper_index = upper_indices[row, col]
            cur_lower_index_plus = lower_indices_plus[row, col]
            cur_upper_index_plus = upper_indices_plus[row, col]

            # Compute the tranform differences to be used later
            lower_transform_diff = transform[row,
                                             cur_lower_index_plus] - transform[row, cur_lower_index]
            upper_transform_diff = transform[row,
                                             cur_upper_index_plus] - transform[row, cur_upper_index]

            # Calculate the alphas for the left and right fractional areas
            alpha_l[row, col] = (lower_limits[row, col]
                                 - transform[row, cur_lower_index]) \
                / lower_transform_diff
            alpha_r[row, col] = (upper_limits[row, col]
                                 - transform[row, cur_upper_index]) \
                / upper_transform_diff

            for cha in range(chan):
                # Compute the full center areas
                center_areas[row, col, cha] = sum_a_table[row, cur_upper_index, cha] \
                    - sum_a_table[row, cur_lower_index_plus, cha]

                # Compute the left and rigth fractional areas. Auxiliar matrices
                # are used to keep the code cleaner and prevent unnecessary processing
                current_pixel = img_padded[row, cur_lower_index, cha]
                current_pixel_plus = img_padded[row, cur_lower_index_plus, cha]
                current_alpha_l = alpha_l[row, col]
                current_alpha_r = alpha_r[row, col]

                left_aux[row, col, cha] = current_pixel \
                    + current_alpha_l * (current_pixel_plus - current_pixel)

                current_pixel = img_padded[row, cur_upper_index, cha]

                right_aux[row, col, cha] = current_pixel \
                    + current_alpha_r * (img_padded[row, cur_upper_index_plus, cha]
                                         - current_pixel)

                left_areas[row, col, cha] = 0.5 * \
                    (left_aux[row, col, cha]
                     + current_pixel_plus) * (1-current_alpha_l) \
                    * lower_transform_diff

                right_areas[row, col, cha] = 0.5 * \
                    (right_aux[row, col, cha]
                     + current_pixel) * current_alpha_r \
                    * upper_transform_diff

    # Perform the summation of the areas, rescaling them to the spacial
    # domain boundaries
    img[:, :, :] = (left_areas[:, :, :] + center_areas[:, :, :] +
                    right_areas[:, :, :]) / (2 * box_radius)

    return img
