""" Main file """
import sys
from time import perf_counter
import cv2 as cv
import functions as fn
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Proper usage: ")
        print(
            "\'python main.py <img_name> <filter_type> <sigma_s> <sigma_r> <iterations> <detail>\'")
        print("To apply detail enhancement, use 0 for false or 1 for true")
        sys.exit()

    # Get parameters
    try:
        img_name = sys.argv[1]
        filter_type = sys.argv[2]
        sigma_s = float(sys.argv[3])
        sigma_r = float(sys.argv[4])
        iterations = int(sys.argv[5])
        detail_enhancement = bool(int(sys.argv[6]))
    except:
        print("Error reading parameters.")
        sys.exit()

    try:
        # Read image
        img = cv.imread(img_name)

       # Start timer
        timer_start = perf_counter()
        print("Start processing...")

        # Apply the filter

        if detail_enhancement:
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            D = np.array(img[:, :, 0]).astype(float) 
            filtered_img = fn.ep_filter(
                img, filter_type, sigma_s, sigma_r, iterations)
            img_n = img / 255
            D[:, :] = img_n[:, :, 0] - filtered_img[:, :, 0]
            D *= 2
            img_n[:, :, 0] += D[:, :]
            filtered_img = cv.convertScaleAbs(img_n * 255)
            filtered_img = cv.cvtColor(filtered_img, cv.COLOR_YCrCb2BGR)
            img = cv.cvtColor(img, cv.COLOR_YCrCb2BGR)
        else:
            filtered_img = fn.ep_filter(
                img, filter_type, sigma_s, sigma_r, iterations)
            filtered_img = cv.convertScaleAbs(filtered_img * 255)

        # Stop timer
        timer_stop = perf_counter()

        # Print time
        time = timer_stop-timer_start
        print("Filter processing time: %.3f " % time, end="seconds.\n")
        sys.stdout.flush()

        # Show images
        cv.imshow('Original image', img)
        if detail_enhancement:
            cv.imshow('Detail Enhanced', filtered_img)
            new_img = 'results/' + \
                str(img_name[:img_name.find('.')]) + \
                '_' + (filter_type) + '_DTE.png'
        else:
            cv.imshow('Filtered image', filtered_img)
            new_img = 'results/' + \
                str(img_name[:img_name.find('.')]) + \
                '_' + (filter_type) + '.png'

        cv.imwrite(new_img, filtered_img)
    except:
        print("Error reading image or incorrect parameters for", "\'"+img_name+"\'")
        sys.exit()

    k = cv.waitKey(0)

    if k == 27:
        cv.destroyAllWindows()
else:
    print("This module cannot be imported.")
