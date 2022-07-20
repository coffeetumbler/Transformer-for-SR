import os
import numpy as np
import random
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def img2tensor(img, normalize=True):
    # handle gray image
    if len(img.shape)==2:
        img = img[:,:,np.newaxis]
    img = np.transpose(img, axes=(2,0,1)).astype(np.float32)
    if normalize:
        img = img / img.max()
    return torch.from_numpy(img)


class dataset_SR(Dataset):
    def __init__(self, df, 
                SR_mode = 96, 
                transforms=None, 
                flip=0.5, 
                rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE],
                degradation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4], 
                channel_wise_noise = True, 
                data='DIV2K',
                setting = "train"):
        super(dataset_SR, self).__init__()
        self.df = df
        self.SR_mode = SR_mode
        self.transforms = transforms
        self.flip = flip
        self.rotation = rotation
        self.data = data
        self.degradation = degradation
        self.channel_wise_noise = channel_wise_noise
        self.setting = setting
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.setting == "train":
            if self.data == 'BSDS200': #DIV2K로 일단 할 거기 때문에 무시
                view_path = '/home/lahj91/SR/SR_training_datasets/BSDS200'+'/'+self.df.iloc[idx]['name']
                origin = cv2.imread(view_path, cv2.IMREAD_UNCHANGED)
                p, q = random.randint(0, origin.shape[0]-self.SR_mode), random.randint(0, origin.shape[1]-self.SR_mode)
                origin = origin[p:p+self.SR_mode,  q:q+self.SR_mode] #random crop(96*96)으로 origin data 생성
                if random.random() < self.flip: #좌우반전
                    origin = cv2.flip(origin, 1)
                if self.rotation != False: #rotation
                    rd = random.random()
                    if rd < 3/4:
                        origin = cv2.rotate(origin, random.choice(self.rotation))
                    else:
                        origin = origin
                degraded = cv2.resize(origin, dsize=(int(self.SR_mode/2),int(self.SR_mode/2)), fx=0.5, fy=0.5, interpolation=random.choice(self.degradation)) 
                interpolated = cv2.resize(degraded, dsize=(self.SR_mode,self.SR_mode), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) #bicubic으로 degraded를 96*96 size 복원
                if self.transforms:
                    for transform in self.transforms:
                        origin = transform(origin)


            if self.data == 'DIV2K':
                view_path = '/home/lahj91/SR/DIV2K/DIV2K_train_HR'+'/'+self.df.iloc[idx]['name']
                origin = cv2.imread(view_path, cv2.IMREAD_UNCHANGED)
                p, q = random.randint(0, origin.shape[0]-self.SR_mode), random.randint(0, origin.shape[1]-self.SR_mode)
                origin = origin[p:p+self.SR_mode,  q:q+self.SR_mode] #random crop(96*96)으로 origin data 생성
                if self.channel_wise_noise: #channel-wise noise
                    pn = np.random.uniform(*[1 - 0.4, 1 + 0.4], size=(3))
                    origin = np.minimum(255., np.maximum(0., origin.transpose(2,0,1) * pn[:,np.newaxis,np.newaxis])).transpose(1,2,0)
                if random.random() < self.flip: #좌우반전
                    origin = cv2.flip(origin, 1)
                if self.rotation != False: #rotation
                    rd = random.random()
                    if rd < 3/4:
                        origin = cv2.rotate(origin, random.choice(self.rotation))
                    else:
                        origin = origin

                degraded = cv2.resize(origin, dsize=(int(self.SR_mode/2),int(self.SR_mode/2)), fx=0.5, fy=0.5, interpolation=random.choice(self.degradation)) #48*48로 random degradation
                interpolated = cv2.resize(degraded, dsize=(self.SR_mode,self.SR_mode), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) #bicubic으로 degraded를 96*96 size 복원
                if self.transforms:
                    for transform in self.transforms:
                        origin = transform(origin)
                        
        elif self.setting == "val":
            if self.data == 'DIV2K':
                view_path = '/home/lahj91/SR/DIV2K/DIV2K_valid_HR'+'/'+self.df.iloc[idx]['name']
                origin = cv2.imread(view_path, cv2.IMREAD_UNCHANGED)
                p, q = random.randint(0, origin.shape[0]-self.SR_mode), random.randint(0, origin.shape[1]-self.SR_mode)
                origin = origin[p:p+self.SR_mode,  q:q+self.SR_mode] #random crop(96*96)으로 origin data 생성
                degraded = cv2.resize(origin, dsize=(int(self.SR_mode/2),int(self.SR_mode/2)), fx=0.5, fy=0.5, interpolation=random.choice(self.degradation)) #48*48로 random degradation
                interpolated = cv2.resize(degraded, dsize=(self.SR_mode,self.SR_mode), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC) #bicubic으로 degraded를 96*96 size 복원
                if self.transforms:
                    for transform in self.transforms:
                        origin = transform(origin)

        return torch.from_numpy(origin.transpose(2,0,1)), torch.from_numpy(degraded.transpose(2,0,1)), torch.from_numpy(interpolated.transpose(2,0,1))
