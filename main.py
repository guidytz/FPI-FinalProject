import sys, os
import numpy as np
import cv2

# img = np.array(cv2.imread('lena.jpg', -1))
# img = img / 255;
# print(img)
# k = cv2.waitKey(0)


# if k == 27:
#     cv2.destroyAllWindows()

def RF(img, sigma_s, sigma_r, n_it=3, joint_img=None):
    I = np.array(img)

    if joint_img != None:
        J = np.array(joint_img)
        if I.shape[:2] != J.shape[:2]:
            print("Input and joint images must have equal width and height!")
            return   
    else:
        J = I

    [h, w, n_channels] = J.shape

    dIcdx = np.diff(J, 1, 2);
    dIcdy = np.diff(J, 1, 1);
    
    dIdx = np.zeros((h, w), dtype=float);
    dIdy = np.zeros((h, w), dtype=float);

    # for c in range(n_channels):
    #     dIdx[1:,:] = dIdx[:,:] + abs( dIcdx[:,:,c] )
    #     # dIdy[:,1:] = dIdy[:,1:] + abs( dIcdy[:,:,c] )

    # dHdx = (1 + sigma_s/sigma_r * dIdx)
    # dVdy = (1 + sigma_s/sigma_r * dIdy)

    dVdy = dVdy.transpose()

    return

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
        print("Nao foi possivel ler a imagem", "\'"+img_name+"\'")
        sys.exit()

    RF(img, 0.1, 0.3)

    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyAllWindows()

    pass