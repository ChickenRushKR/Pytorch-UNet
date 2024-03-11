import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from openpyxl import load_workbook
import math
import cv2

line = 1
t = 0

all_dir = 'D:/khj/dataset/isens/All'
y_min_cut = 350
y_max_cut = 850
x_min_cut = 2000
x_max_cut = 3200


def main():
    dir_list = os.listdir(all_dir)
    datamap = open('./datamap.txt','w')
    save_dir = all_dir + '/result/'
    try:
        os.mkdir(save_dir)
    except:
        print(save_dir, 'is already exist.')
    save_dir1 = save_dir + '/org/'
    try:
        os.mkdir(save_dir1)
    except:
        print(save_dir1, 'is already exist.')
    save_dir2 = save_dir + '/cat/'
    try:
        os.mkdir(save_dir2)
    except:
        print(save_dir2, 'is already exist.')
    save_dir3 = save_dir + '/filt/'
    try:
        os.mkdir(save_dir3)
    except:
        print(save_dir3, 'is already exist.')
    dataidx = 1
    for data in dir_list:
        # global line
        # global t
        # t = 0
        name = all_dir + '/' + data
        
        print(f'{dataidx}.png ---------- {name} ')
        # continue
        img_gray = cv2.imread(name, 0)       # 파일 읽기 (grayscale)
        if img_gray is None:
            print("image open error: ", name)
            exit()
        img_gray = img_gray[:-100, x_min_cut:x_max_cut]
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        img_gray_blur = cv2.GaussianBlur(img_gray, (0, 0), 2)  # 블러처리 (자르기용도)
        ret, step1 = cv2.threshold(img_gray_blur, 210, 0, cv2.THRESH_TOZERO_INV) # 배경 자르기 (step 1)

        # try:
        contours, hierarchy = cv2.findContours(image=step1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) # 배경자르고 contour
        # draw contours on the original image
        image_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        image_bgr_copy = image_bgr.copy()
        idxs = []
        for i in range(len(contours)):
            idxs.append(len(contours[i]))
        idx = np.argmax(idxs)
        rect = cv2.minAreaRect(contours[idx])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image=image_bgr, contours=contours, contourIdx=-1, color=(10, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.drawContours(image_bgr,[box],contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow('firstcontour', image_bgr)
        centroid = (box[0] + box[1] + box[2] + box[3]) / 4
        # centroid = np.int0(centroid)
        y_list = box[:][:,1]
        y_top1 = box[:][y_list.argsort()[3]]
        y_top2 = box[:][y_list.argsort()[2]]
        distance = math.dist(y_top1, y_top2)
        x_diff = y_top1[0] - y_top2[0]
        theta = math.acos(x_diff/distance)
        degree = np.rad2deg(theta)
        if degree < 90:
            degree += 180
        matrix = cv2.getRotationMatrix2D((centroid[0], centroid[1]), degree-180, 1.0)
        image_bgr_rot = cv2.warpAffine(image_bgr_copy, matrix, (image_bgr_copy.shape[1], image_bgr_copy.shape[0]))
        
        box_rot = np.zeros((4,2))
        for i in range(4):
            x_p = math.cos(theta) * (box[i][0]-centroid[0]) + math.sin(theta) * (box[i][1]-centroid[1])
            y_p = -math.sin(theta) * (box[i][0]-centroid[0]) + math.cos(theta) * (box[i][1]-centroid[1])
            x_p += centroid[0]
            y_p += centroid[1]
            x_p = int(x_p)
            y_p = int(y_p)
            box_rot[i][0] = x_p
            box_rot[i][1] = y_p
        # box_rot = np.asarray(box_rot)
        box_rot = np.int0(box_rot)
        cr_x_min = box_rot[:,0].min()
        cr_x_max = box_rot[:,0].max()
        cr_y_min = box_rot[:,1].min()
        cr_y_max = box_rot[:,1].max()
        
        # i = 0
        # for pt in box_rot:
            # cv2.circle(image_bgr_rot, (pt[0],pt[1]), 15, color=(0, 0, 255), thickness=-1)
            # cv2.putText(image_bgr_rot, f'pt{i}', (pt[0],pt[1]), cv2.FONT_ITALIC, 2, color=(0, 0, 0), thickness=3)
            # i+=1
        # cv2.circle(image_bgr_rot, (int(centroid[0]),int(centroid[1])), 15, color=(0, 0, 0), thickness=-1)
        image_bgr_rot_crop = image_bgr_rot[360:872,cr_x_max-50:cr_x_max+14]
        if image_bgr_rot_crop.shape[0] != 512 or image_bgr_rot_crop.shape[1] != 64:
            print(image_bgr_rot_crop.shape, "ERROR SIZE DETECTED")
            continue
            # cv2.imshow('firstcontour', image_bgr)
            # cv2.imshow('rotate', image_bgr_rot)
            # if cv2.waitKey(40000) == ord('n'):
            #     continue
            # cv2.destroyAllwindows()
        else:
            # print(image_bgr_rot_crop.shape)
            image_bgr_down = cv2.resize(image_bgr, (0,0), fx=0.3, fy=0.3)
            image_bgr_rot_down = cv2.resize(image_bgr_rot, (0,0), fx=0.3, fy=0.3)
            # cv2.imshow('firstcontour', image_bgr_down)
            # cv2.imshow('rotate', image_bgr_rot_down)
            im = image_bgr_rot_crop
            f_img = cv2.Scharr(im, -1, 1, 0)
            f_rev_img = cv2.bitwise_not(f_img)
            res = cv2.add(im * 0.88, f_rev_img * 0.12)
            # cv2.imwrite(save_dir1 + str(dataidx) + '.png', image_bgr_rot_crop)
            # cv2.imwrite(save_dir2 +  str(dataidx) + '.png', res)
            # cv2.imwrite(save_dir3 +  str(dataidx) + '.png', f_rev_img)
            datamap.write(f'{dataidx}.png\t{name}\n')
        # cv2.imshow('crop1', image_bgr_rot_crop)
        # cv2.imwrite(save_dir + data + '.png', image_bgr_rot_crop)
        dataidx += 1
        # if cv2.waitKey(40000) == ord('n'):
        #     continue
        # if cv2.waitKey(40000) == ord('b'):
        #     data -= 2
        #     continue
    datamap.close()

if __name__=='__main__':
    main()