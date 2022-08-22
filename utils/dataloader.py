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

class valid_batch_making:
    def __init__(self, items, batch=4):
        self.current = 0
        self.items = items
        self.stop = len(items['degraded'].keys())
        self.batch = batch

        concatenated = torch.tensor([])
        for key, value in self.items['degraded'].items():
            concatenated = torch.cat((value, concatenated))
        self.items = concatenated

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.stop:
            temp = self.items[self.current:min(self.current + self.batch, self.stop), :, :, :]
            self.current += self.batch
            return temp
        else:
            raise StopIteration

class dataset_SR(Dataset):
    def __init__(self,
                setting = "train",
                augmentation = True,
                SR_mode = 2,
                data="DIV2K",
                data_merge = False,
                Matlab_mode = True,
                Matlab_degraded_mode = "bicubic"
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
        self.intersection = config.PIXEL_INTERSECTION
        self.Matlab_mode = Matlab_mode
        self.degraded_mode = Matlab_degraded_mode
        self.Matlab_mode_path = config.MATLAB_MODE_PATH

        if self.Matlab_mode == True:
            if self.setting == "train":
                self.DATA_LIST = config.MATLAB_TRAINING_DATA_LIST
            else:
                self.DATA_LIST = config.MATLAB_TEST_DATA_LIST
            if self.data_merge:
                data_all_HR = []
                data_all_degraded = []
                for i in self.DATA_LIST:
                    df = pd.read_csv(config.MATLAB_DATA_LIST_DIR + i + '_' + self.setting + '_HR'+ '.csv').sort_values(by='name')
                    data_all_HR.append(df)
                    df_degraded = pd.read_csv(config.MATLAB_DATA_LIST_DIR + i + '_' + self.setting + '_LR_bicubic_X' + str(self.SR_mode) + '.csv').sort_values(by='name')
                    data_all_degraded.append(df_degraded)
                self.data_name_list_HR = pd.concat(data_all_HR, axis=0, ignore_index=True)
                self.data_name_list_degraded = pd.concat(data_all_degraded, axis=0, ignore_index=True)
            else:
                self.data_name_list_HR = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_' + self.setting + "_HR.csv").sort_values(by='name')
                self.data_name_list_degraded = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_' + self.setting + "_LR_bicubic_X" + str(self.SR_mode) + '.csv').sort_values(by='name')


        else:
            if self.setting == "train":
                self.DATA_LIST = config.TRAINING_DATA_LIST
            else:
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
        if self.Matlab_mode == True:
            return len(self.data_name_list_HR)
        else:
            return len(self.data_name_list)
    

    def __getitem__(self, idx):

        if self.Matlab_mode and (self.setting == "train"):
            view_path_HR = self.Matlab_mode_path[self.setting] + self.data_name_list_HR.iloc[idx]['data'] + '_train_HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.Matlab_mode_path[self.setting] + self.data_name_list_degraded.iloc[idx]['data'] + '_train_LR_bicubic/X' + str(self.SR_mode) + '/' + self.data_name_list_degraded.iloc[idx]['name']
            HR_image = cv2.imread(view_path_HR, cv2.IMREAD_UNCHANGED).copy()
            degraded_image = cv2.imread(view_path_degraded, cv2.IMREAD_UNCHANGED).copy()
            p, q = random.randint(0, degraded_image.shape[0]-48), random.randint(0, degraded_image.shape[1]-48)
            HR_image = HR_image[2*p:2*(p+48),  2*q:2*(q+48)]
            degraded_image = degraded_image[p:p+48,  q:q+48]
            interpolated = cv2.resize(degraded_image, dsize=(self.img_size,self.img_size), interpolation=cv2.INTER_CUBIC)#와꾸만 맞출려고
            items = {}
            items['origin'] = torch.from_numpy(HR_image.transpose(2,0,1)).float() / 255
            items['degraded'] = self.normalize_img(torch.from_numpy(degraded_image.transpose(2,0,1)).float() / 255)
            items['interpolated'] = self.normalize_img(torch.from_numpy(interpolated.transpose(2,0,1)).float() / 255)
            return items
        # elif or(self.Matlab_mode == True, self.setting == "test", self.setting == "valid"):

        else:
            if self.data_merge:
                view_path = self.MODE_PATH[self.setting] + self.data_name_list.iloc[idx]['data'] + '/' + self.data_name_list.iloc[idx]['name']
            else:
                view_path = self.prefix + self.data_name_list.iloc[idx]['name']
                
            origin = cv2.imread(view_path, cv2.IMREAD_UNCHANGED).copy()
            if self.setting == "valid":
                idx_x = [i for i in range(0,origin.shape[0]-self.img_size+1, self.img_size-self.intersection)]
                # idx_x_end = idx_x + self.img_size
                if origin.shape[0]%(self.img_size) != 0:
                    idx_x = np.append(idx_x, origin.shape[0]-self.img_size)
                idx_y = [i for i in range(0,origin.shape[1]-self.img_size+1, self.img_size-self.intersection)]
                if origin.shape[1]%(self.img_size) != 0:
                    idx_y = np.append(idx_y, origin.shape[1]-self.img_size)
                item_degraded = {}
                mask = torch.zeros(origin.shape[0], origin.shape[1])
                for i in idx_x:
                    for j in idx_y:
                        item_degraded[(i,j)] = self.normalize_img(torch.from_numpy(cv2.resize(origin[i:i+self.img_size, j:j+self.img_size], dsize=(48,48), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)).float() / 255)
                        mask[i:i+self.img_size, j:j+self.img_size] += 1
                origin = torch.from_numpy(origin.transpose(2,0,1)).float() / 255
                items = {"origin" : origin, "degraded" : item_degraded, "mask" : mask}
                return items




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
    if setting !='valid':    
        if setting == 'train':
            augmentation = True
        elif setting == 'test':
            augmentation = False
        dataloader = dataset_SR(setting=setting, augmentation=augmentation, **kwargs)
        return DataLoader(dataloader, batch_size=batch_size, shuffle=augmentation, pin_memory=pin_memory, num_workers=num_workers)
    elif setting == 'valid':

        # augmentation = False
        # batch_size = 1
        # dataloader = dataset_SR(setting=setting, augmentation=augmentation, **kwargs)
        # loader = DataLoader(dataloader, batch_size=batch_size=1, shuffle=augmentation, pin_memory=pin_memory, num_sorkers=num_workers)
        dataloader = dataset_SR(setting=setting, augmentation=False, **kwargs)
        return DataLoader(dataloader, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
