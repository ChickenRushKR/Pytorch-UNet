import cv2
import os

img_dir = 'result/sample_img/'
pred_dir = 'result/sample_pred/'
true_dir = 'result/sample_true/'
concat_dir = 'result/concat/'

img_list = os.listdir(img_dir)
pred_list = os.listdir(pred_dir)
true_list = os.listdir(true_dir)

for i in range(len(img_list)):
    img_gray = cv2.imread(img_dir + img_list[i])       # 파일 읽기 (grayscale)
    mask_gray = cv2.imread(pred_dir + pred_list[i])       # 파일 읽기 (grayscale)
    true_gray = cv2.imread(true_dir + true_list[i])       # 파일 읽기 (grayscale)
    true_gray *= 255
    concat = cv2.add(img_gray, mask_gray)
    concat2 = cv2.add(img_gray, true_gray)
    print('pred:', img_list[i], pred_list[i])
    print('true:', img_list[i], true_list[i])
    cv2.imwrite(concat_dir + 'pred_' + img_list[i], concat)
    cv2.imwrite(concat_dir + 'true_' + img_list[i], concat2)