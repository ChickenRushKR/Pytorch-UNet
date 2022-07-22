import numpy as np
from PIL import Image
import os
import cv2

dir = './data/test_data2/'
ml = os.listdir(dir)
for data in ml:
    im = cv2.imread(dir+data)
    f_img = cv2.Scharr(im, -1, 1, 0)
    # arr = np.zeros((512, 64))
    # for y in range(64):
    #     for x in range(64):
    #         if im[x][y][0] == 0:
    #             arr[x][y] = 0
    #         else:
    #             arr[x][y] = 1
    cv2.imwrite('./data/test_data2_scharr/'+data, f_img)
    # print(f_img.shape)
    print(data)