import cv2
import os

img_dir = 'result/test_data2_concat/'
pred_dir = 'result/pred/'
true_dir = 'result/test_mask3/'
concat_dir = 'result/concat/'

img_list = os.listdir(img_dir)
pred_list = os.listdir(pred_dir)
true_list = os.listdir(true_dir)

for i in range(len(img_list)):
    img_gray = cv2.imread(img_dir + img_list[i])       # 파일 읽기 (grayscale)
    mask_gray = cv2.imread(pred_dir + pred_list[i])       # 파일 읽기 (grayscale)
    true_gray = cv2.imread(true_dir + true_list[i])       # 파일 읽기 (grayscale)
    true_gray *= 255
    concat = cv2.add(img_gray * 0.7, mask_gray * 0.3)
    concat2 = cv2.add(img_gray * 0.7, true_gray * 0.3)
    print('pred:', img_list[i], pred_list[i])
    print('true:', img_list[i], true_list[i])
    filename = img_list[i].split('.')[0]
    cv2.imwrite(concat_dir + filename + '_pred.png', concat)
    cv2.imwrite(concat_dir + filename + '_true.png', concat2)