import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from tqdm import tqdm
from retinaface import RetinaFace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
new_img_dir = '/opt/ml/input/data/train/new_imgs'
img_path = '/opt/ml/input/data/train/images'

cnt = 0
crop_range = 70
    
for paths in tqdm(os.listdir(img_path)):
    if paths[0] == '.': continue
    
    sub_dir = os.path.join(img_path, paths)
    
    for imgs in os.listdir(sub_dir):
        if imgs[0] == '.': continue
        
        img_dir = os.path.join(sub_dir, imgs)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)
        
        if probs[0]:
            xmin = int(boxes[0, 0])-crop_range
            ymin = int(boxes[0, 1])-crop_range
            xmax = int(boxes[0, 2])+crop_range-10
            ymax = int(boxes[0, 3])+crop_range-10
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
        else:
            result_detected = RetinaFace.detect_faces(img_dir)
            if type(result_detected) == dict:
                ymin=result_detected['face_1']['facial_area'][1]-crop_range
                ymax=result_detected['face_1']['facial_area'][3]+crop_range-10
                xmin=result_detected['face_1']['facial_area'][0]-crop_range
                xmax=result_detected['face_1']['facial_area'][2]+crop_range-10
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > 384: xmax = 384
                if ymax > 512: ymax = 512
            else:
                xmin = 100
                xmax = 400
                ymin = 50
                ymax = 350
        
        img = img[ymin:ymax, xmin:xmax, :]
            
        img = cv2.resize(img,(300,300))
        
        tmp = os.path.join(new_img_dir, paths)
        cnt += 1
        plt.imsave(os.path.join(tmp, imgs), img)
        
new_img_dir = '/opt/ml/input/data/eval/new_imgs'
img_path = '/opt/ml/input/data/eval/images'

cnt = 0

print("not yet")

for imgs in tqdm(os.listdir(img_path)):
    # print(imgs)
    if imgs[0] == '.': continue
        
    img_dir = os.path.join(img_path, imgs)
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #mtcnn 적용
    boxes,probs = mtcnn.detect(img)
        
        # boxes 확인
        # if len(probs) > 1: 
            # print(boxes)
    if probs[0]:
        xmin = int(boxes[0, 0])-crop_range
        ymin = int(boxes[0, 1])-crop_range
        xmax = int(boxes[0, 2])+crop_range-10
        ymax = int(boxes[0, 3])+crop_range-10
            
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > 384: xmax = 384
        if ymax > 512: ymax = 512
    else:
        result_detected = RetinaFace.detect_faces(img_dir)
        if type(result_detected) == dict:
            ymin=result_detected['face_1']['facial_area'][1]-crop_range
            ymax=result_detected['face_1']['facial_area'][3]+crop_range-10
            xmin=result_detected['face_1']['facial_area'][0]-crop_range
            xmax=result_detected['face_1']['facial_area'][2]+crop_range-10
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
        else:
            xmin = 100
            xmax = 400
            ymin = 50
            ymax = 350
        
            
    img = img[ymin:ymax, xmin:xmax, :]
        
            
    img = cv2.resize(img,(300,300))
    
    plt.imsave(os.path.join(new_img_dir, imgs), img)
        
print("done")