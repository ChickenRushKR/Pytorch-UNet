import cv2
import os

img_dir = 'result/12241314/data/'
pred_dir = 'result/12241314/pred/'
# true_dir = 'result/12201651/true/'
concat_dir = 'result/12241314/cat/'

img_list = os.listdir(img_dir)
pred_list = os.listdir(pred_dir)
# true_list = os.listdir(true_dir)
# os.mkdir(concat_dir)
for i in range(len(img_list)):
    img_gray = cv2.imread(img_dir + img_list[i], 0)       # 파일 읽기 (grayscale)
    mask_gray = cv2.imread(pred_dir + pred_list[i], 0)       # 파일 읽기 (grayscale)
    ret, mask_gray = cv2.threshold(mask_gray, 30, 255, cv2.THRESH_BINARY)
    mask_gray = cv2.GaussianBlur(mask_gray, (0, 0), 1)
    # print(type(mask_gray))
    # true_gray = cv2.imread(true_dir + true_list[i], 0)       # 파일 읽기 (grayscale)
    # true_gray *= 255
    concat = cv2.add(img_gray * 0.7, mask_gray * 0.3)
    # concat2 = cv2.add(img_gray * 0.7, true_gray * 0.3)
    # print(img_list[i], pred_list[i], true_list[i])
    # print('true:', img_list[i], true_list[i])
    # if pred_list[i] != true_list[i]:
    #     print("name is different", pred_list[i], true_list[i])
    #     break
    filename = img_list[i].split('.')[0]
    print(filename)
    cv2.imwrite(concat_dir + filename + '_pred.png', concat)
    # cv2.imwrite(concat_dir + filename + '_true.png', concat2)