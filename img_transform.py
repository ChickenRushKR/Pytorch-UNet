import numpy as np
from PIL import Image
import os
import cv2

dir = './data/outdated/test_data2/'
ml = os.listdir(dir)
for data in ml:
    print(dir + data)
    im = cv2.imread(dir+data, cv2.IMREAD_GRAYSCALE)
    f_img = cv2.Scharr(im, -1, 1, 0)
    f_rev_img = cv2.bitwise_not(f_img)
    res = cv2.add(im * 0.85, f_rev_img * 0.15)
    # arr = np.zeros((512, 64))
    # for y in range(512):
        # for x in range(64):
            # if im[y, x] != 0:
                # arr[y, x] = 1
            # else:
                # arr[y, x] = 0
            # print(arr[y, x])
    cv2.imwrite('./data/data2_concat3/'+data, res)
    
    # print(im.shape)