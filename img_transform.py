import numpy as np
from PIL import Image
import os
import cv2

dir = './data/test_data/'
ml = os.listdir(dir)
for data in ml:
    im = cv2.imread(dir+data)
    f_img = cv2.Scharr(im, -1, 1, 0)
    cv2.imwrite('./data/test_data2/'+data, f_img)
    # print(f_img.shape)
    print(data)