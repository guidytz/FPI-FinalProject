import math
import sys
import cv2
import numpy as np
import functions as fn
import filters as ft


def main():
    img = cv2.imread('statue.png')

    sigma_s = 200
    sigma_r = 0.3
    iterations = 3
    ep_filter = 'RF'

    # Normalize the image
    img_norm = img/255

    # Get the transformed signal for use in the filters
    # In the RF filter, we do not need to integrate the domain transform because
    # it uses the derivatives directly
    if ep_filter == 'RF':
        [hor_differences, ver_differences] = fn.domain_transform(
            img_norm, sigma_s, sigma_r, False)
    else:
        [hor_transform, ver_transform] = fn.domain_transform(
            img_norm, sigma_s, sigma_r, True)

    # Initialize the H sigma to be used next
    sigma_h = sigma_s

    # Initialize the output image
    img_out = img_norm

    # Aplly the choosen filter
    for i in range(iterations-1):
        # Calculate the current sigma H using equation 14 of the paper
        cur_sigma_h = sigma_h * \
            math.sqrt(3) * (2**(iterations-(i+1))) / \
            math.sqrt(4**iterations - 1)

        # If the filter is not RF, we'll need the filter_radius
        if ep_filter != 'RF':
            filter_radius = math.sqrt(3) * cur_sigma_h

        # Apply the filter
        if ep_filter == 'RF':
            img_out = ft.recursive_filtering(
                img_out, hor_differences, cur_sigma_h)
        elif ep_filter == 'IC':
            img_out = None
        else:
            img_out = None

        # Transpose the imagem so we can apply the filter vertically
        img_out = fn.image_transpose(img_out)

        if ep_filter == 'RF':
            img_out = ft.recursive_filtering(
                img_out, np.transpose(ver_differences), cur_sigma_h)
        elif ep_filter == 'IC':
            img_out = None
        else:
            img_out = None

        img_out = fn.image_transpose(img_out)

    cv2.imshow('Imagem filtrada', img_out)

    key = cv2.waitKey(0)

    if key == 27:
        cv2.destroyAllWindows()


main()
