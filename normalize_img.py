import numpy as np
from PIL import Image
import os
import cv2

ml = os.listdir('./data/test_data/')
for data in ml:
    im = cv2.imread('./data/test_data/'+data)
    im_norm = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)
    cv2.imwrite('./data/test_norm/'+data, im_norm)
    print(data)