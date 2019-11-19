""" Main file """
import sys
from time import perf_counter
import cv2
import functions as fn

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Proper usage: ")
        print(
            "\'python main.py <img_name> <filter_type> <sigma_s> <sigma_r> <iterations>\'")
        sys.exit()

    # Get parameters
    try:
        img_name = sys.argv[1]
        filter_type = sys.argv[2]
        sigma_s = float(sys.argv[3])
        sigma_r = float(sys.argv[4])
        iterations = int(sys.argv[5])
    except:
        print("Error reading parameters.")
        sys.exit()
        
    # Read image
    img = cv2.imread(img_name)

    try:
        # Start timer
        timer_start = perf_counter()
        print("Start processing...")

        # Apply the filter
        filtered_img = fn.ep_filter(img, filter_type, sigma_s, sigma_r, iterations)

        # Stop timer
        timer_stop = perf_counter()

        # Print time
        time = timer_stop-timer_start
        print("Filter processing time: %.3f " % time, end="seconds.")
        sys.stdout.flush()

        # Show images
        cv2.imshow('Original image', img)
        cv2.imshow('Filtered image', filtered_img)
    except:
        print("Error reading image or incorrect parameters for", "\'"+img_name+"\'")
        sys.exit()

    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyAllWindows()
else:
    print("This module cannot be imported.")
