import argparse
import logging
import os

import pandas as pd
import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from datetime import datetime as dt

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        masks_pred, points_pred = net(img)        # net output mask and heatmap

        if net.n_classes > 1:
            probs = F.softmax(masks_pred, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
        

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(), points_pred


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--inputdir', '-id', metavar='INPUT', help='Filenames of input images dir')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(in_files):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return list(map(_generate_name, in_files))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def gaussian(xL, yL, sigma, H, W):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

def convertToHM(img, keypoints, sigma=5):

        W = img.size[0]
        H = img.size[1]
        
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

        for i in range(0, nKeypoints // 2):
            x = keypoints[i * 2]
            y = keypoints[1 + 2 * i]

            channel_hm = gaussian(x, y, sigma, H, W)

            img_hm[:, :, i] = channel_hm
        
        # img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints // 2, 1))

        return img_hm

def maskToKeypoints(mask):
    # mask = np.reshape(mask, newshape=(96,96))
    kp = np.unravel_index(np.argmax(mask, axis=None), shape=(512,64))
    return kp[1], kp[0]

def calcKeypoints(img, kps_pred):
    # kps_gt = []
    # kps_preds = []
    # nbatches = len(gen)

    # for i in range(nbatches+1):
        # print("\nBatch {}".format(i))
    # imgs, batch_gt = gen[i]
    # batch_preds = model.predict_on_batch(imgs)
    # print(batch_gt.shape)
    # print(batch_preds.shape)
    # n_imgs = imgs.shape[0]
    # print("\t# of Images {}".format(n_imgs))
    # for j in range(n_imgs):

    mask_pred = kps_pred.cpu().numpy()
    mask_pred = np.reshape(mask_pred, newshape=(512, 64, 2))
    # nchannels = mask_gt.shape[-1]
    # print(nchannels)
    pred_list = []

    for k in range(2):
        xpred, ypred = maskToKeypoints(mask_pred[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])

        pred_list.append(xpred)
        pred_list.append(ypred)
    
    return np.array(pred_list, dtype=np.float32)

def show_keypoints(batch_imgs, predictions=None):

    def draw_keypoints(img, keypoints, col):
        # print("\n{}".format(len(keypoints)))
        for i in range(0, len(keypoints)-1, 2):
            # print(i)
            kpx = int(keypoints[i])
            kpy = int(keypoints[i+1])
            img = cv2.circle(img, center=(kpx,kpy), radius=2, color=col, thickness=2)

        return img

    img = batch_imgs
    img = np.reshape(img, newshape=(512, 64, 3))
    # img = np.stack([img,img,img], axis=-1)

    # draw ground-truth keypoints on image
    # draw predicted keypoints on image
    if predictions is not None:
        img = draw_keypoints(img, predictions, col=(255,0,0))
    
    return img

if __name__ == '__main__':
    args = get_args()
    # in_files = args.input
    in_files = os.listdir(args.inputdir)
    out_files = get_output_filenames(in_files)
    
    now = dt.now()
    time_now = now.strftime("%m%d%H%M")
    print(time_now)
    out_dir = './result/' + time_now
    try:
        os.mkdir(out_dir)
        print(out_dir, 'was created.')
    except:
        print(out_dir, 'is already existed')
        exit()
    out_pred_dir = './result/' + time_now + '/pred/'
    try:
        os.mkdir(out_pred_dir)
        print(out_pred_dir, 'was created.')
    except:
        print(out_pred_dir, 'is already existed')
        exit()
    out_cat_dir = './result/' + time_now + '/cat/'
    try:
        os.mkdir(out_cat_dir)
        print(out_cat_dir, 'was created.')
    except:
        print(out_cat_dir, 'is already existed')
        exit()
    
    excel = pd.read_csv('hm_result.csv')
    net = UNet(n_channels=3, n_classes=4, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')
    print('Model loaded!')

    for i, filename in enumerate(in_files):

        logging.info(f'\nPredicting image {args.inputdir + filename} ...')
        # print(f'Predicting image {filename} ...')
        gt_pts = excel[excel['name']==filename].values[0][1:]
        keypoints = [gt_pts[0],gt_pts[1]]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        
        img = Image.open(args.inputdir + filename)
        img_gray = Image.open(args.inputdir + filename).convert("L")
        
        mask, pred_point = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # heatmap = convertToHM(img, keypoints)
        # kps_preds = calcKeypoints(img, pred_point)

        if not args.no_save:
            # kpt_result = show_keypoints(img, predictions=kps_preds)
            
            # cv2.imwrite(out_pred_dir + filename.split('.')[0] + 'kpt.' + filename.split('.')[1], kpt_result)
            out_filename = out_files[i]
            result = mask_to_image(mask)

            result.save(out_pred_dir + filename)

            imgnp=np.array(img)  
            imgcv=cv2.cvtColor(imgnp, cv2.COLOR_RGB2GRAY)

            resultcv=np.array(result)  
            # resultcv=cv2.cvtColor(resultnp, cv2.COLOR_RGB2GRAY)

            ret, resultcv = cv2.threshold(resultcv, 30, 255, cv2.THRESH_BINARY)
            resultcv = cv2.GaussianBlur(resultcv, (0, 0), 1)

            thickness = 0
            
            for y in range(0,int(keypoints[1]) + 10):
                for x in range(64):
                    resultcv[y][x] = 0
            
            for y in range(int(keypoints[1]) + 10, int(keypoints[1]) + 150):
                for x in range(64):
                    if resultcv[y][x] > 200:
                        thickness += 1
            print(filename, thickness / 140 * 1.5)
            for y in range(int(keypoints[1]) + 150, 512):
                for x in range(64):
                    resultcv[y][x] = 0
            # resultcv = cv2.GaussianBlur(resultcv, (0, 0), 1)
            
            concat = cv2.add(imgcv * 0.7, resultcv * 0.3)
            cv2.imwrite(out_cat_dir + filename, concat)

            logging.info(f'Mask saved to {out_pred_dir + filename}')
            
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)