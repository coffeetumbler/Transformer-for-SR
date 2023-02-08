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
from image_processing import imresize_np

class valid_batch_making:
    def __init__(self, items, batch=16):
        self.current = 0
        self.items = items
        self.stop = len(items['degraded'].keys())
        self.batch = batch

        concatenated = torch.tensor([])
        concatenated_keys = []
        for key, value in self.items['degraded'].items():
            concatenated = torch.cat((concatenated, value))
            concatenated_keys.append(key)

        self.items = concatenated
        self.keys = concatenated_keys
    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.stop:
            batched_degraded_img = self.items[self.current:min(self.current + self.batch, self.stop), :, :, :]
            keys = self.keys[self.current:min(self.current + self.batch, self.stop)]
            self.current += self.batch
            return keys, batched_degraded_img
        else:
            raise StopIteration

            
class dataset_SR(Dataset):
    def __init__(self,
                 setting = "train",
                 augmentation = True,
                 channel_wise_noise = 1/2,
                 SR_mode = 2,
                 data="DIV2K",
                 data_merge = False,
                 n_patch_in_img = 4,
                 lr_img_size=48
                 ):
        super(dataset_SR, self).__init__()
        self.SR_mode = SR_mode
        self.lr_img_size = lr_img_size
        self.img_size = SR_mode * lr_img_size
        
        self.data = data
        self.setting = setting
        self.setting_index = 0 if (setting == 'train') else (1 if (setting == 'valid') else (2 if (setting == 'evaluation') else 3))
        
        self.augmentation = augmentation
        self.channel_wise_noise = channel_wise_noise
        self.flip = 0.5
        self.rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        
        self.n_patch_in_img = n_patch_in_img
        self.normalize_img = Normalize(config.IMG_NORM_MEAN, config.IMG_NORM_STD)
        self.data_merge = data_merge
        
        self.intersection = config.PIXEL_INTERSECTION
        self.Matlab_mode_path = config.MATLAB_MODE_PATH
    
        if self.setting_index == 0:
            self.DATA_LIST = config.MATLAB_TRAINING_DATA_LIST
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
                self.data_name_list_degraded = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_' + self.setting + "_LR_bicubic_X" + str(self.SR_mode) + ".csv").sort_values(by='name')

        elif (self.setting_index == 2) or (self.setting_index == 3):
            self.DATA_LIST = config.MATLAB_EVALUATION_DATA_LIST
            self.data_name_list_HR = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_valid_ipt_HR.csv').sort_values(by='name')
            self.data_name_list_degraded = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_valid_ipt_LR_bicubic_X' + str(self.SR_mode) + ".csv").sort_values(by='name')

        elif self.setting_index == 1:
            self.DATA_LIST = config.MATLAB_VALID_DATA_LIST
            if self.data_merge:
                data_all_HR = []
                data_all_degraded = []
                for i in self.DATA_LIST:
                    df = pd.read_csv(config.MATLAB_DATA_LIST_DIR + i + '_valid_ipt_HR.csv').sort_values(by='name')
                    data_all_HR.append(df)
                    df_degraded = pd.read_csv(config.MATLAB_DATA_LIST_DIR + i + '_valid_ipt_LR_bicubic_X' + str(self.SR_mode) + '.csv').sort_values(by='name')
                    data_all_degraded.append(df_degraded)
                self.data_name_list_HR = pd.concat(data_all_HR, axis=0, ignore_index=True)
                self.data_name_list_degraded = pd.concat(data_all_degraded, axis=0, ignore_index=True)
            else:
                self.data_name_list_HR = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_valid_ipt_HR.csv').sort_values(by='name')
                self.data_name_list_degraded = pd.read_csv(config.MATLAB_DATA_LIST_DIR + self.data + '_valid_ipt_LR_bicubic_X' + str(self.SR_mode) + ".csv").sort_values(by='name')


    def __len__(self):
        return len(self.data_name_list_HR)
    

    def __getitem__(self, idx):
        if self.setting_index == 0:
            view_path_HR = self.Matlab_mode_path[self.setting] + self.data_name_list_HR.iloc[idx]['data'] + '_train_HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.Matlab_mode_path[self.setting] + self.data_name_list_degraded.iloc[idx]['data'] + '_train_LR_bicubic/X' + str(self.SR_mode) + '/' + self.data_name_list_degraded.iloc[idx]['name']

            HR_image = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
            degraded_image = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()
            
            # single image augmentation
            # p, q = random.randint(0, degraded_image.shape[0]-48), random.randint(0, degraded_image.shape[1]-48)
            # HR_image = HR_image[self.SR_mode*p:self.SR_mode*(p+48),  self.SR_mode*q:self.SR_mode*(q+48)]
            # degraded_image = degraded_image[p:p+48,  q:q+48]

            # if self.augmentation:
            #     if random.random() < self.flip: #좌우반전
            #         HR_image = cv2.flip(HR_image, 1)
            #         degraded_image = cv2.flip(degraded_image, 1)

            #     if random.random() < 3/4:
            #         choice = random.choice(self.rotation)
            #         HR_image = cv2.rotate(HR_image, choice)
            #         degraded_image = cv2.rotate(degraded_image, choice)

            #     if random.random() < self.channel_wise_noise: #channel-wise noise
            #         pn = np.random.uniform(*[1 - 0.4, 1 + 0.4], size=(3))
            #         HR_image = np.round(np.minimum(255., np.maximum(0., HR_image * pn[np.newaxis, np.newaxis, :])))
            #         degraded_image = np.round(np.minimum(255., np.maximum(0., degraded_image * pn[np.newaxis, np.newaxis, :])))
            # items = {}
            # items['origin'] = torch.from_numpy(HR_image.transpose(2,0,1)).float() / 255
            # items['degraded'] = self.normalize_img(torch.from_numpy(degraded_image.transpose(2,0,1)).float() / 255)
            # return items

            # multiple image augmentation
            p = np.random.randint(0, degraded_image.shape[0]-self.lr_img_size, size=(self.n_patch_in_img))
            q = np.random.randint(0, degraded_image.shape[1]-self.lr_img_size, size=(self.n_patch_in_img))
            
            HR_images = []
            degraded_images = []

            for indices in np.stack((p, p+self.lr_img_size, q, q+self.lr_img_size), axis=-1):
                hr_indices = indices * self.SR_mode
                HR_images.append(HR_image[hr_indices[0]:hr_indices[1], hr_indices[2]:hr_indices[3]])
                degraded_images.append(degraded_image[indices[0]:indices[1], indices[2]:indices[3]])
            
            # flip and rotation
            for i, probs, choice in zip(range(self.n_patch_in_img), np.random.rand(self.n_patch_in_img, 2), np.random.choice(self.rotation, self.n_patch_in_img)):
                # flip in 1/2 probability
                if probs[0] < self.flip:
                    HR_images[i] = cv2.flip(HR_images[i], 1)
                    degraded_images[i] = cv2.flip(degraded_images[i], 1)
                # rotation
                if probs[1] < 0.75:
                    HR_images[i] = cv2.rotate(HR_images[i], choice)
                    degraded_images[i] = cv2.rotate(degraded_images[i], choice)
                
            HR_image = np.stack(HR_images)  # (self.n_patch_in_img, 48*SR_mode, 48*SR_mode, 3)
            degraded_image = np.stack(degraded_images)  # (self.n_patch_in_img, 48, 48, 3)
            
            # channel-wise noise
#             pn = np.random.uniform(*[1 - 0.4, 1 + 0.4], size=(self.n_patch_in_img, 3))
#             pn[np.random.rand(self.n_patch_in_img) >= self.channel_wise_noise] = 1
#             HR_image = np.round(np.minimum(255., np.maximum(0., HR_image * pn[:, np.newaxis, np.newaxis])))
#             degraded_image = np.round(np.minimum(255., np.maximum(0., degraded_image * pn[:, np.newaxis, np.newaxis])))
            
            # color shifting
            pn = np.random.randint(-10, 10, size=(self.n_patch_in_img, 3))
            pn[np.random.rand(self.n_patch_in_img) >= self.channel_wise_noise] = 0
            HR_image = np.minimum(255, np.maximum(0, HR_image.astype(int) + pn[:, np.newaxis, np.newaxis]))
            degraded_image = np.minimum(255, np.maximum(0, degraded_image.astype(int) + pn[:, np.newaxis, np.newaxis]))

            items = {}
            items['origin'] = torch.from_numpy(HR_image.transpose(0,3,1,2)).float() / 255
            items['degraded'] = self.normalize_img(torch.from_numpy(degraded_image.transpose(0,3,1,2)).float() / 255)
            
            return items

        elif self.setting_index == 1:
            view_path_HR = self.Matlab_mode_path[self.setting] + self.data_name_list_HR.iloc[idx]['data'] + '/HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.Matlab_mode_path[self.setting] + self.data_name_list_degraded.iloc[idx]['data'] + '/LR_bicubic/X' + str(self.SR_mode) + '/' + self.data_name_list_degraded.iloc[idx]['name']

            HR_image = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
            degraded_image = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()

            p = np.random.randint(0, degraded_image.shape[0]-self.lr_img_size, size=(self.n_patch_in_img))
            q = np.random.randint(0, degraded_image.shape[1]-self.lr_img_size, size=(self.n_patch_in_img))
            
            HR_images = []
            degraded_images = []
            interpolated_images = []

            for indices in np.stack((p, p+self.lr_img_size, q, q+self.lr_img_size), axis=-1):
                hr_indices = indices * self.SR_mode
                HR_images.append(HR_image[hr_indices[0]:hr_indices[1], hr_indices[2]:hr_indices[3]])
                degraded_images.append(degraded_image[indices[0]:indices[1], indices[2]:indices[3]])
                interpolated_images.append(imresize_np(degraded_images[-1], self.SR_mode))

            HR_image = np.stack(HR_images)  # (self.n_patch_in_img, 48*SR_mode, 48*SR_mode, 3)
            degraded_image = np.stack(degraded_images)  # (self.n_patch_in_img, 48, 48, 3)
            interpolated_image = np.stack(interpolated_images) # (self.n_patch_in_img, 48*SR_mode, 48*SR_mode, 3)

            items = {}
            items['origin'] = torch.from_numpy(HR_image.transpose(0,3,1,2)).float() / 255
            items['degraded'] = self.normalize_img(torch.from_numpy(degraded_image.transpose(0,3,1,2)).float() / 255)
            items['interpolated'] = torch.from_numpy(interpolated_image.transpose(0,3,1,2)).float() / 255
            return items

        elif self.setting_index == 2:
            view_path_HR = self.Matlab_mode_path[self.setting] + self.data_name_list_HR.iloc[idx]['data'] + '/HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.Matlab_mode_path[self.setting] + self.data_name_list_degraded.iloc[idx]['data'] + '/LR_bicubic/X' + str(self.SR_mode) + '/' + self.data_name_list_degraded.iloc[idx]['name']

            origin = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
            origin = origin[0:(origin.shape[0]//self.SR_mode)*self.SR_mode, 
                            0:(origin.shape[1]//self.SR_mode)*self.SR_mode]
            degraded = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()

            idx_x = [i for i in range(0,origin.shape[0]-self.img_size+1, self.img_size-self.intersection)]
            # idx_x_end = idx_x + self.img_size
            if idx_x[-1] != (origin.shape[0] - self.img_size):
                idx_x = np.append(idx_x, origin.shape[0]-self.img_size)

            idx_y = [i for i in range(0,origin.shape[1]-self.img_size+1, self.img_size-self.intersection)]
            if idx_y[-1] != (origin.shape[1] - self.img_size):
                idx_y = np.append(idx_y, origin.shape[1]-self.img_size)

            item_degraded = {}
            mask = torch.zeros(origin.shape[0], origin.shape[1])

            for i in idx_x:
                for j in idx_y:
                    item_degraded[(i,j)] = self.normalize_img(torch.from_numpy(degraded[i // self.SR_mode : (i+self.img_size) // self.SR_mode,
                                                                                        j // self.SR_mode : (j+self.img_size) // self.SR_mode].transpose(2,0,1)).float() / 255)
                    mask[i:i+self.img_size, j:j+self.img_size] += 1

            origin = torch.from_numpy(origin.transpose(2,0,1)).float() / 255
            items = {"origin" : origin, "degraded" : item_degraded, "mask" : mask}
            return items

        elif self.setting_index == 3:
            view_path_HR = self.Matlab_mode_path[self.setting] + self.data_name_list_HR.iloc[idx]['data'] + '/HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.Matlab_mode_path[self.setting] + self.data_name_list_degraded.iloc[idx]['data'] + '/LR_bicubic/X' + str(self.SR_mode) + '/' + self.data_name_list_degraded.iloc[idx]['name']
            
            HR_image = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
            HR_image = HR_image[0:(HR_image.shape[0]//self.SR_mode)*self.SR_mode, 
                                0:(HR_image.shape[1]//self.SR_mode)*self.SR_mode]
            HR_image = torch.from_numpy(HR_image.transpose(2,0,1)).float() / 255
            
            degraded_image = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()
            degraded_image = self.normalize_img(torch.from_numpy(degraded_image.transpose(2,0,1)).float() / 255)
            items = {"origin" : HR_image, "degraded" : degraded_image}
            return items
        
        
        
def get_dataloader(batch_size=16, setting='train', augmentation=True, pin_memory=False, num_workers=0, **kwargs): #num_workers는 hyperparameter tunning의 영역
    if setting == 'train':    
        augmentation = True
    elif setting == "evaluation":
        augmentation = False
        batch_size = 1
    elif setting == "valid":
        augmentation = False
    else:
        aumentation = False
        batch_size = 1
    dataloader = dataset_SR(setting=setting, augmentation=augmentation, **kwargs)
    return DataLoader(dataloader, batch_size=batch_size, shuffle=augmentation, pin_memory=pin_memory, num_workers=num_workers)
