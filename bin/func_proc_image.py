#!/usr/bin/env python
# coding: utf-8

import numpy as np
from mtcnn_cv2 import MTCNN
import cv2
from IPython.display import display, Image

import warnings
warnings.filterwarnings('ignore')


def display_cv_image(image, format='.png'):
    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
    display(Image(data=decoded_bytes))


# resize image limitted max size (fixed aspect)
def scale_to_size(img, max_size):
    h, w = img.shape[:2]
    
    if h>w:
        width = round(w * (max_size / h)) # *0.25) * 4
        height = max_size
    else:
        width = max_size
        height = round(h * (width / w)) # *0.25) * 4
    
    #width = round(w * (height / h))
    #height = round(h * (width / w))
    dst = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    return dst


# create campus and and arrange image
def fill_image(img, width, height):
    # width x height の3レイヤー(BGR)を定義
    size = height, width, 3
    dst = np.zeros(size, dtype=np.uint8)
    
    h, w = img.shape[:2]
    y0 = round((height - h)*0.5)
    y1 = y0 + h
    x0 = round((width - w)*0.5)
    x1 = x0 + w
    
    dst[y0:y1, x0:x1, :] = img[:, :, :]
    
    return dst


def split_image_per_face(img, base_size=224, th_conf=0.85):
    #img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    #img = cv2.imread(fp)

    img_src = img.copy()
    
    detector = MTCNN()
    res = detector.detect_faces(img)
    #print(res)
    
    result_list = []
    
    if len(res)<1:
        return result_list

    for i in range(len(res)):
        box = res[i]["box"]
        conf = res[i]["confidence"]
        if conf < th_conf:
            continue

        tmp_list = []
        
        src = img_src[box[1] : box[1] + box[3], box[0] : box[0]+box[2]]
        dst = scale_to_size(src, base_size)
        dst2 = fill_image(dst, base_size, base_size)
        img2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        
        tmp_list.append(img2)
        tmp_list.append(round(box[0] + 0.5*box[2]))
        tmp_list.append(round(box[1] + 0.5*box[3]))
        tmp_list.append(conf)
        
        result_list.append(tmp_list)

        cv2.rectangle(img,
                      (box[0], box[1]),
                      (box[0]+box[2], box[1] + box[3]),
                      (0,155,255),
                      2)
        #cv2.imwrite("result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("result.jpg", img)

    return result_list


def split_image_per_face_pad(img, base_size=224, th_conf=0.85, pad=0.1, back=False):
    #img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    #img = cv2.imread(fp)

    img_src = img.copy()
    
    detector = MTCNN()
    res = detector.detect_faces(img)
    #print(res)
    
    result_list = []
    h, w = img.shape[:2]
    
    if len(res)<1:
        return result_list

    for i in range(len(res)):
        box = res[i]["box"]
        conf = res[i]["confidence"]
        if conf < th_conf:
            continue

        tmp_list = []
        
        cx = round(box[0] + 0.5*box[2])
        cy = round(box[1] + 0.5*box[3])
        
        crop_w = box[2]
        crop_h = box[3]
        
        if back == True:
            if crop_w > crop_h:
                crop_h = crop_w
            else:
                crop_w = crop_h
        
        x0 = cx-round(crop_w*(1+pad*2)*0.5)
        if x0 <0:
            x0=0
        y0 = cy-round(crop_h*(1+pad*2)*0.5)
        if y0 <0:
            y0=0
        x1 = cx+round(crop_w*(1+pad*2)*0.5)
        if x1 > w-1:
            x1=w-1
        y1 = cy+round(crop_h*(1+pad*2)*0.5)
        if y1 > h-1:
            y1=h-1
        
        src = img_src[y0 : y1, x0 : x1].copy()
        
        if back == True:
            dst2 = cv2.resize(src, dsize=(base_size, base_size), interpolation=cv2.INTER_CUBIC)
        else:
            dst = scale_to_size(src, base_size)
            dst2 = fill_image(dst, base_size, base_size)
        
        img2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        
        tmp_list.append(img2)
        tmp_list.append(cx)
        tmp_list.append(cy)
        tmp_list.append(conf)
        
        result_list.append(tmp_list)

        cv2.rectangle(img,
                      (x0, y0),
                      (x1, y1),
                      (0,155,255),
                      2)
        #cv2.imwrite("result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("result.jpg", img)
        #dst = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    return result_list

def split_image_per_face_pad_circle(img, base_size=224, th_conf=0.85, pad=0.1, back=False):
    #img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    #img = cv2.imread(fp)

    img_src = img.copy()
    
    detector = MTCNN()
    res = detector.detect_faces(img)
    #print(res)
    
    result_list = []
    h, w = img.shape[:2]
    
    if len(res)<1:
        return result_list

    for i in range(len(res)):
        box = res[i]["box"]
        conf = res[i]["confidence"]
        if conf < th_conf:
            continue

        tmp_list = []
        
        cx = round(box[0] + 0.5*box[2])
        cy = round(box[1] + 0.5*box[3])
        
        crop_w = box[2]
        crop_h = box[3]
        
        if back == True:
            if crop_w > crop_h:
                crop_h = crop_w
            else:
                crop_w = crop_h
        
        x0 = cx-round(crop_w*(1+pad*2)*0.5)
        if x0 <0:
            x0=0
        y0 = cy-round(crop_h*(1+pad*2)*0.5)
        if y0 <0:
            y0=0
        x1 = cx+round(crop_w*(1+pad*2)*0.5)
        if x1 > w-1:
            x1=w-1
        y1 = cy+round(crop_h*(1+pad*2)*0.5)
        if y1 > h-1:
            y1=h-1
        
        src = img_src[y0 : y1, x0 : x1].copy()
        
        # マスク作成 (黒く塗りつぶす画素の値は0)
        mask_h = src.shape[0]
        mask_w = src.shape[1]
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        
        # 円を描画する関数 circle() を利用してマスクの残したい部分を 255 にしている。
        #cv2.circle(mask, center=(mask_h // 2, mask_w // 2), radius=150, color=255, thickness=-1)
        cv2.ellipse(mask, ((mask_w // 2, mask_h // 2), (mask_w, mask_h), 0), color=255, thickness=-1)
        
        src[mask==0] = [0, 0, 0]  # mask の値が 0 の画素は黒で塗りつぶす。
        
        if back == True:
            dst2 = cv2.resize(src, dsize=(base_size, base_size), interpolation=cv2.INTER_CUBIC)
        else:
            dst = scale_to_size(src, base_size)
            dst2 = fill_image(dst, base_size, base_size)
        
        img2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        
        tmp_list.append(img2)
        tmp_list.append(cx)
        tmp_list.append(cy)
        tmp_list.append(conf)
        
        result_list.append(tmp_list)

        """
        cv2.rectangle(img,
                      (x0, y0),
                      (x1, y1),
                      (0,155,255),
                      2)
        """
        #cv2.imwrite("result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("result.jpg", img)
        #dst = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    return result_list

def split_image_per_face_pad_circle_min(img, base_size=224, th_conf=0.85, pad=0.1, back=False, min_size=2500):
    #img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    #img = cv2.imread(fp)

    img_src = img.copy()
    
    detector = MTCNN()
    res = detector.detect_faces(img)
    #print(res)
    
    result_list = []
    h, w = img.shape[:2]
    
    if len(res)<1:
        return result_list

    for i in range(len(res)):
        box = res[i]["box"]
        conf = res[i]["confidence"]
        if conf < th_conf:
            continue

        _size = box[3] * box[2]
        if _size < min_size:
            continue
            
        tmp_list = []
        
        cx = round(box[0] + 0.5*box[2])
        cy = round(box[1] + 0.5*box[3])
        
        crop_w = box[2]
        crop_h = box[3]
        
        if back == True:
            if crop_w > crop_h:
                crop_h = crop_w
            else:
                crop_w = crop_h
        
        x0 = cx-round(crop_w*(1+pad*2)*0.5)
        if x0 <0:
            x0=0
        y0 = cy-round(crop_h*(1+pad*2)*0.5)
        if y0 <0:
            y0=0
        x1 = cx+round(crop_w*(1+pad*2)*0.5)
        if x1 > w-1:
            x1=w-1
        y1 = cy+round(crop_h*(1+pad*2)*0.5)
        if y1 > h-1:
            y1=h-1
        
        src = img_src[y0 : y1, x0 : x1].copy()
        
        # マスク作成 (黒く塗りつぶす画素の値は0)
        mask_h = src.shape[0]
        mask_w = src.shape[1]
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        
        # 円を描画する関数 circle() を利用してマスクの残したい部分を 255 にしている。
        #cv2.circle(mask, center=(mask_h // 2, mask_w // 2), radius=150, color=255, thickness=-1)
        cv2.ellipse(mask, ((mask_w // 2, mask_h // 2), (mask_w, mask_h), 0), color=255, thickness=-1)
        
        src[mask==0] = [0, 0, 0]  # mask の値が 0 の画素は黒で塗りつぶす。
        
        if back == True:
            dst2 = cv2.resize(src, dsize=(base_size, base_size), interpolation=cv2.INTER_CUBIC)
        else:
            dst = scale_to_size(src, base_size)
            dst2 = fill_image(dst, base_size, base_size)
        
        img2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        
        tmp_list.append(img2)
        tmp_list.append(cx)
        tmp_list.append(cy)
        tmp_list.append(conf)
        
        result_list.append(tmp_list)

        """
        cv2.rectangle(img,
                      (x0, y0),
                      (x1, y1),
                      (0,155,255),
                      2)
        """
        #cv2.imwrite("result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite("result.jpg", img)
        #dst = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    return result_list

def split_image_per_face_min(img, base_size=224, th_conf=0.85, min_size=0):
    #img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    #img = cv2.imread(fp)

    img_src = img.copy()
    
    detector = MTCNN()
    res = detector.detect_faces(img)
    #print(res)
    
    result_list = []
    
    if len(res)<1:
        return result_list

    for i in range(len(res)):
        box = res[i]["box"]
        conf = res[i]["confidence"]
        if conf < th_conf:
            continue

        tmp_list = []
        
        src = img_src[box[1] : box[1] + box[3], box[0] : box[0]+box[2]]
        
        _size = box[3] * box[2]
        
        if _size >= min_size:        
            dst = scale_to_size(src, base_size)
            dst2 = fill_image(dst, base_size, base_size)
            img2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)

            tmp_list.append(img2)
            tmp_list.append(round(box[0] + 0.5*box[2]))
            tmp_list.append(round(box[1] + 0.5*box[3]))
            tmp_list.append(conf)

            result_list.append(tmp_list)

            cv2.rectangle(img,
                          (box[0], box[1]),
                          (box[0]+box[2], box[1] + box[3]),
                          (0,155,255),
                          2)
            #cv2.imwrite("result.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            #cv2.imwrite("result.jpg", img)

    return result_list


# HSV H(色相)の変更
def changedH(hsvimage, shift):
    #hsvimage = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
    hi = hsvimage[:,:,0].astype(np.int32)
    if shift < 0:
        nhi = hi.flatten()
        for px in nhi:
            if px < 0:
                px = 255 - px
        nhi = nhi.reshape(hsvimage.shape[:2])
        hi = nhi.astype(np.uint8)
    chimg = (hi + shift) % 255
    hsvimage[:,:,0] = chimg
    hsv8 = hsvimage.astype(np.uint8)
    return hsv8


# HSV S(彩度),V(明度)の変更
def changedSV(hsvimage, alpha, beta, color_idx):
    #hsvimage = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
    hsvf = hsvimage.astype(np.float32)
    hsvf[:,:,color_idx] = np.clip(hsvf[:,:,color_idx] * alpha+beta, 0, 255)
    hsv8 = hsvf.astype(np.uint8)
    return hsv8 #cv2.cvtColor(hsv8,cv2.COLOR_HSV2BGR_FULL)

# HSV S(彩度)の変更
def changedS(hsvimage, alpha, beta):
    return changedSV(hsvimage, alpha, beta, 1)

# HSV V(明度)の変更
def changedV(hsvimage, alpha, beta):
    return changedSV(hsvimage, alpha, beta, 2)


def crop_image(img):
    # to grayscale
    img_copy = img.copy()
    img_copy[img_copy!=0]=255
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    x1_min,x2_max,y1_min,y2_max=0,299,0,299
    
    for i in range(150):
        if np.sum(gray[:,i])!=0:
            x1_min=i
            break;
    for i in range(299, 150, -1):
        if np.sum(gray[:,i])!=0:
            x2_max=i
            break;
    for i in range(150):
        if np.sum(gray[i,:])!=0:
            y1_min=i
            break;
    for i in range(299, 150, -1):
        if np.sum(gray[i,:])!=0:
            y2_max=i
            break;

    #cv2.rectangle(img, (x1_min, y1_min), (x2_max, y2_max), (0, 255, 0), 3)
    crop_img = img[y1_min:y2_max, x1_min:x2_max]
    
    return crop_img

