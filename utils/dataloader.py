import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

import config



class dataset_SR(Dataset):
    def __init__(self,
                setting = "train",
                augmentation = True,
                SR_mode = 2,
                data="DIV2K",
                data_merge = False
                ):
        super(dataset_SR, self).__init__()
        self.SR_mode = SR_mode
        self.img_size = SR_mode * 48
        self.data = data
        self.setting = setting
        
        self.augmentation = augmentation
        self.flip = 0.5
        self.rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        self.degradation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        self.channel_wise_noise = True
        
        self.MODE_PATH = config.MODE_PATH
        self.normalize_img = Normalize(config.IMG_NORM_MEAN, config.IMG_NORM_STD)
        self.prefix = self.MODE_PATH[self.setting] + self.data + '/'
        self.data_merge = data_merge

        if self.setting == "train":
            self.DATA_LIST = config.TRAINING_DATA_LIST
        elif self.setting == "test":
            self.DATA_LIST = config.TEST_DATA_LIST
        
        if self.data_merge:
            data_all = []
            for i in self.DATA_LIST:
                df = pd.read_csv(config.DATA_LIST_DIR + self.setting + "_" + i + ".csv")
                data_all.append(df)
            self.data_name_list = pd.concat(data_all, axis=0, ignore_index=True)
        else:
            self.data_name_list = pd.read_csv(config.DATA_LIST_DIR + self.setting + "_" + self.data + ".csv")
            

    def __len__(self):
        return len(self.data_name_list)
    

    def __getitem__(self, idx):
        if self.data_merge:
            view_path = self.MODE_PATH[self.setting] + self.data_name_list.iloc[idx]['data'] + '/' + self.data_name_list.iloc[idx]['name']
        else:
            view_path = self.prefix + self.data_name_list.iloc[idx]['name']
            
        origin = cv2.imread(view_path, cv2.IMREAD_UNCHANGED).copy()
        p, q = random.randint(0, origin.shape[0]-self.img_size), random.randint(0, origin.shape[1]-self.img_size)
        origin = origin[p:p+self.img_size,  q:q+self.img_size] #random crop(96*96)으로 origin data 생성
        if self.augmentation:
            if self.channel_wise_noise: #channel-wise noise
                pn = np.random.uniform(*[1 - 0.4, 1 + 0.4], size=(3))
                origin = np.minimum(255., np.maximum(0., origin * pn[np.newaxis, np.newaxis, :]))
            if random.random() < self.flip: #좌우반전
                origin = cv2.flip(origin, 1)
            if self.rotation != False: #rotation
                rd = random.random()
                if rd < 3/4:
                    origin = cv2.rotate(origin, random.choice(self.rotation))
            degraded = cv2.resize(origin, dsize=(48,48), interpolation=random.choice(self.degradation))
            
        else:
            degraded = cv2.resize(origin, dsize=(48,48), interpolation=cv2.INTER_CUBIC)
            
        interpolated = cv2.resize(degraded, dsize=(self.img_size,self.img_size), interpolation=cv2.INTER_CUBIC) #bicubic으로 degraded를 96*96 size 복원 fx=0.5, fy=0.5,

        items = {}
        items['origin'] = torch.from_numpy(origin.transpose(2,0,1)).float() / 255
        items['degraded'] = self.normalize_img(torch.from_numpy(degraded.transpose(2,0,1)).float() / 255)
        items['interpolated'] = self.normalize_img(torch.from_numpy(interpolated.transpose(2,0,1)).float() / 255)

        return items

    
    
def get_dataloader(batch_size=16, setting='train', augmentation=True, pin_memory=False, num_workers=0, **kwargs): #num_workers는 hyperparameter tunning의 영역
    if setting == 'train':
        augmentation = True
    elif setting == 'test':
        augmentation = False
    elif setting == 'valid':
        setting = 'test'
    dataloader = dataset_SR(setting=setting, augmentation=augmentation, **kwargs)
    return DataLoader(dataloader, batch_size=batch_size, shuffle=augmentation, pin_memory=pin_memory, num_workers=num_workers)
