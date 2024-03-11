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

dir_list = os.listdir(all_dir)
excel = pd.read_excel('id_map.xlsx', sheet_name='labelmap')
dataidx = 401


for data in dir_list:
        # global line
        # global t
        # t = 0
        
        rowdata = excel[excel['idxname']==f'{dataidx}.png']
        # print(rowdata)
        rowvalues = rowdata.values[0][2:-2]
        if dataidx >= 303:
            rowvalues = rowvalues[:4]
        # print(rowvalues)
        name = all_dir + '/' + rowdata['origname'].values[0]
        print(f'{dataidx}.png ---------- {name} ')
        # continue
        img_gray = cv2.imread(name, 0)       # 파일 읽기 (grayscale)
        if img_gray is None or img_gray.shape[0] == 0:
            print("image open error: ", name)
            exit()
        # else:
            # img=cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        xpoints = [int(rowvalues[i]) for i in range(0, len(rowvalues), 2)]
        ypoints = [int(rowvalues[i]) for i in range(1, len(rowvalues), 2)]
        xp_max = np.max(xpoints)
        xp_min = np.min(xpoints)
        yp_max = np.max(ypoints)
        yp_min = np.min(ypoints)
        print(xpoints, ypoints)
        print(xp_max, xp_min)
        print(yp_max, yp_min)
        if yp_min < 300 or xp_min < 300:
            img_gray_crop = img_gray
            img_crop=cv2.cvtColor(img_gray_crop, cv2.COLOR_GRAY2RGB)
            for i in range(len(xpoints)):
                cv2.circle(img_crop, (xpoints[i],ypoints[i]), 3, color=(0, 0, 255), thickness=-1)
            img_crop = img_crop[yp_min:yp_max,xp_min:xp_max]
        else:
            img_gray_crop = img_gray[yp_min-300:yp_max+300,xp_min-300:xp_max+300]
            img_crop=cv2.cvtColor(img_gray_crop, cv2.COLOR_GRAY2RGB)
            for i in range(len(xpoints)):
                cv2.circle(img_crop, (xpoints[i]-xp_min+300,ypoints[i]-yp_min+300), 3, color=(0, 0, 255), thickness=-1)
            # cv2.putText(img_crop, f'pt{i}', (xpoints[i]-xp_min+280,ypoints[i]-yp_min+280), cv2.FONT_ITALIC, 1, color=(0, 0, 0), thickness=2)

        ''' 예측결과 같이 띄우기 '''
        # predname = './result/12261732/cat/' + rowdata['idxname'].values[0]
        # img_pred = cv2.imread(predname, 0)       # 파일 읽기 (grayscale)
        # img_pred = cv2.resize(img_pred, (0,0), fx=1.5, fy=1.5)
        # img_crop = cv2.resize(img_crop, (0,0), fx=1.5, fy=1.5)

        cv2.imwrite(f"./data/label/{dataidx}.png", img_crop)
        # cv2.imshow('prediction', img_pred)
        # cv2.imshow('isens_label', img_crop)
        dataidx += 1
        # if cv2.waitKey(10000) == ord('n'):
            # continue