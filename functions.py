"""File containg auxiliar functions for applying the filters."""

import math
import sys
import numpy as np
import filters as ft


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


def limits_indices(transform, hgt, wdt, radius):
    """ This function returns the indices of the pixels associated with the
    upper and lower limits of a box kernel"""

    # Get the lower and upper limits of the kernel
    lower_limits = transform - radius
    upper_limits = transform + radius

    # Initialize the pixels matrices
    lower_indices = np.zeros(np.array(transform).shape)
    upper_indices = np.zeros(np.array(transform).shape)

    for row in range(hgt):
        # We create auxiliar matrices to reduce unnecessary processing and to keep
        # the code clear
        transform_row = np.concatenate([transform[row, :], np.array([np.inf])])

        lower_limits_row = lower_limits[row, :]
        upper_limits_row = upper_limits[row, :]

        lower_indices_row = np.zeros([wdt])
        upper_indices_row = np.zeros([wdt])

        lower_indices_row[0] = (
            transform_row > lower_limits_row[0]).nonzero()[0][0]
        upper_indices_row[0] = (
            transform_row > upper_limits_row[0]).nonzero()[0][0]

        for col in range(1, wdt):
            prev_lower_indices_row = lower_indices_row[col-1]
            prev_upper_indices_row = upper_indices_row[col-1]
            lower_indices_row[col] = prev_lower_indices_row + (
                transform_row[int(prev_lower_indices_row):] > lower_limits_row[col]).nonzero()[0][0]
            upper_indices_row[col] = prev_upper_indices_row + (
                transform_row[int(prev_upper_indices_row):] > upper_limits_row[col]).nonzero()[0][0]

        lower_indices[row, :] = lower_indices_row
        upper_indices[row, :] = upper_indices_row

        upper_indices = upper_indices.astype(int)
        lower_indices = lower_indices.astype(int)

    return (lower_indices, upper_indices)


def ep_filter(img, filter_type, sigma_s, sigma_r, iterations):
    """Main function for applying the filters"""

    # Normalize the image
    img_norm = img/255

    # Get the transformed signal for use in the filters
    # In the RF filter, we do not need to integrate the domain transform because
    # it uses the derivatives directly
    if filter_type == 'RF':
        [hor_differences, ver_differences] = domain_transform(
            img_norm, sigma_s, sigma_r, False)
    else:
        [hor_transform, ver_transform] = domain_transform(
            img_norm, sigma_s, sigma_r, True)

    # Initialize the H sigma to be used next
    sigma_h = sigma_s

    # Initialize the output image
    img_out = img_norm

    progress = iterations * 2
    step = 100 / progress
    elapsed = step

    # Aplly the choosen filter
    for i in range(iterations):
        # Calculate the current sigma H using equation 14 of the paper
        cur_sigma_h = sigma_h * \
            math.sqrt(3) * (2**(iterations-(i+1))) / \
            math.sqrt(4**iterations - 1)

        # Apply the filter
        if filter_type == 'RF':
            img_out = ft.recursive_filtering(
                img_out, hor_differences, cur_sigma_h)
        elif filter_type == 'IC':
            img_out = ft.interpolated_convolution(
                img_out, hor_transform, cur_sigma_h)
        elif filter_type == 'NC':
            img_out = ft.normalized_convolution(
                img_out, hor_transform, cur_sigma_h)
        else:
            raise ValueError("Unknown filter specified")

        # Transpose the imagem so we can apply the filter vertically
        img_out = image_transpose(img_out)

        progress -= 1
        print("%.0f" % elapsed, end="%...")
        elapsed += step
        sys.stdout.flush()

        if filter_type == 'RF':
            img_out = ft.recursive_filtering(
                img_out, np.transpose(ver_differences), cur_sigma_h)
        elif filter_type == 'IC':
            img_out = ft.interpolated_convolution(
                img_out, np.transpose(ver_transform), cur_sigma_h)
        else:
            img_out = ft.normalized_convolution(
                img_out, np.transpose(ver_transform), cur_sigma_h)

        # Transpose it back
        img_out = image_transpose(img_out)

        progress -= 1
        print("%.0f" % elapsed, end="%...")
        elapsed += step
        sys.stdout.flush()

    print()
    return img_out
