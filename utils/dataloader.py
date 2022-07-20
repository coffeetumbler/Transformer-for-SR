import os
import numpy as np
import random
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

class dataset_SR(Dataset):
    def __init__(self,
                SR_mode = 2,
                flip=0.5, 
                rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE],
                degradation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4], 
                channel_wise_noise = True, 
                normalization = True,
                data='DIV2K',
                setting = "train"):
        super(dataset_SR, self).__init__()
        self.SR_mode = SR_mode
        self.flip = flip
        self.rotation = rotation
        self.data = data
        self.degradation = degradation
        self.channel_wise_noise = channel_wise_noise
        self.normalization = normalization
        self.setting = setting
        self.mode_path = {"train" : '/home/lahj91/SR/SR_training_datasets', "val" : '/home/lahj91/SR/SR_testing_datasets'}
        self.data_name_list = pd.read_csv('./DataName/' + self.setting + "_" + self.data + ".csv")
        self.normalize_img = Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    def __len__(self):
        return len(self.data_name_list)
    def __getitem__(self, idx):
        view_path = self.mode_path[self.setting]+ '/'+ self.data + '/' + self.data_name_list.iloc[idx]['name']
        origin = cv2.imread(view_path, cv2.IMREAD_UNCHANGED)
        p, q = random.randint(0, origin.shape[0]-self.SR_mode*48), random.randint(0, origin.shape[1]-self.SR_mode*48)
        origin = origin[p:p+self.SR_mode*48,  q:q+self.SR_mode*48] #random crop(96*96)으로 origin data 생성
        if self.channel_wise_noise: #channel-wise noise
            pn = np.random.uniform(*[1 - 0.4, 1 + 0.4], size=(3))
            origin = np.minimum(255., np.maximum(0., origin * pn[np.newaxis, np.newaxis, :]))
        if random.random() < self.flip: #좌우반전
            origin = cv2.flip(origin, 1)
        if self.rotation != False: #rotation
            rd = random.random()
            if rd < 3/4:
                origin = cv2.rotate(origin, random.choice(self.rotation))
            else:
                origin = origin
        degraded = cv2.resize(origin, dsize=(int(self.SR_mode*24),int(self.SR_mode*24)), fx=0.5, fy=0.5, interpolation=random.choice(self.degradation)) 
        interpolated = cv2.resize(degraded, dsize=(self.SR_mode*48,self.SR_mode*48), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) #bicubic으로 degraded를 96*96 size 복원
        return torch.from_numpy(origin.transpose(2,0,1))/255, self.normalize_img(torch.from_numpy(degraded.transpose(2,0,1))/255), self.normalize_img(torch.from_numpy(interpolated.transpose(2,0,1))/255)