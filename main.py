import sys
import os
import numpy as np
import cv2


def main():
    """Função principal para execução do programa"""

    if __name__ == "__main__":
        if len(sys.argv) != 2:
            print("Proper usage: ")
            print("\'python main.py <img_name>\'")
            sys.exit()

        [img_name] = sys.argv[1:]
        img = cv2.imread(img_name, -1)

        try:
            cv2.imshow('image', img)
        except:
            print("Error reading image", "\'"+img_name+"\'")
            sys.exit()

        k = cv2.waitKey(0)

        if k == 27:
            cv2.destroyAllWindows()

    print("This module cannot be imported.")
