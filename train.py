import os, sys
import time
#from pickletools import optimize
sys.path.append(os.path.dirname(os.path.abspath('./train.ipynb')))
import numpy as np
import random
import cv2
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models import transformer
from models import whole_models
from models import submodels
#dataloader의 정확한 위치를 어디로 하실꺼죠? 그거에 따라서 좀 조절할필요가 있을 거 같아요! 
from utils.dataloader import get_dataloader
from utils import config
from utils import image_processing

from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
import easydict

#Basic Setting : 임의로 정했습니다.
args = easydict.EasyDict({
        "gpu" : 0,
        "learing_rate": 1e-4,
        "batch_size" : 16,
        "num_epochs" : 100,
        "loss" : 'l1',
        "weight_decay" : 1e-4,
        "upscale" : 2,
        "Optimizer" : "AdamW",
        "CosineLRScheduler":'''[lr_min=args.learing_rate*0.01,
        \t\twarmup_lr_init=args.learing_rate*0.001,
        \t\twarmup_t=max_iter//10,cycle_limit=1,
        \t\tt_in_epochs=False]'''
    })

#Date
now=time.localtime()
ntime="%04d%02d%02d_%02d%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

#Save Path
save_path ='./logs'
save_path_upscale = save_path+'/x'+str(args.upscale)
save_path_date = save_path_upscale+'/'+ntime
save_path_state_dict = save_path_date+'/state_dict'
save_path_model = save_path_date+'/model'
save_path_output = save_path_date+'/output'

if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(save_path_upscale):
    os.mkdir(save_path_upscale)
if not os.path.exists(save_path_date):
    os.mkdir(save_path_date)
else: #To prevent overwrite. Ex. A, A(0), A(1), ...
    for i in range(500):
        save_path_date=save_path_date+"("+str(i)+")"
        if not os.path.exists(save_path_date):
            os.mkdir(save_path_date)
            break
        save_path_date = save_path_upscale+'/'+ntime
if not os.path.exists(save_path_state_dict):
    os.mkdir(save_path_state_dict)
if not os.path.exists(save_path_model):
    os.mkdir(save_path_model)
if not os.path.exists(save_path_output):
    os.mkdir(save_path_output)

#hyperparameters save
file = open(save_path_date+"/hyperparameters.txt", "w")
for code, name in args.items():
    file.write(f'{code} : {name}\n')
'''file.write(str(args))'''
file.close()

#GPU setting 
cudnn.benchmark=True #input size stable -> Good
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

#Model setting & save
model = whole_models.SRTransformer(upscale=args.upscale)
model.to(device)
torch.save(model, os.path.join(save_path_model+'model.pt'))

#Loss setting
if args.loss == 'l1':
    criterion = nn.L1Loss().to(device)
elif args.loss == 'mse':
    criterion = nn.MSELoss().to(device)

#optimizer : 임의로 정했습니다.
optimizer = optim.AdamW(params = model.parameters(), lr=args.learing_rate, weight_decay = args.weight_decay)

#Dataset
dataloader_train = get_dataloader(setting ="train", num_workers=4)
train_size = len(dataloader_train)
max_iter = (train_size//args.batch_size + 1) * args.num_epochs

dataloader_val= get_dataloader(setting ="valid", num_workers=4)
val_size = len(dataloader_val)

#scheduler : warmup + cosin lr decay : 임의로 정했습니다.
scheduler = CosineLRScheduler(optimizer,
                              t_initial=max_iter,
                              lr_min=args.learing_rate*0.01,
                              warmup_lr_init=args.learing_rate*0.001,
                              warmup_t=max_iter//10,
                              cycle_limit=1,
                              t_in_epochs=False
                            )

#For validation report
min_valid_loss = 1000
min_valid_epoch = 0
_iter = 0

def PSNR(A, B):
    length=len(A)
    psnr=0
    for i in range(length):
        mse=nn.MSELoss().to(device)(A[i], B[i])
        psnr+=(-10)*torch.log10(mse)
    return psnr

#Training & Validation
for epoch in range(args.num_epochs):

    #Train
    model.train()
    
    with tqdm(len(train_size)) as t:
        t.set_description('epoch : {}/{}'.format(epoch, args.num_epochs-1))

        for batch, items in enumerate(dataloader_train) :
            n_batch = items['origin'].size()[0]

            items['origin'].to(device)
            items['degraded'].to(device)
            items['interpolated'].to(device)

            mid_result=model(items['degraded'], items['interpolated'])
            mid_result_denorm=image_processing.denormalize(mid_result, device=device) #device cpu 해 놓은 이유!
            loss = criterion(mid_result, items['origin'])


            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.1)

            optimizer.step()

            _iter+=1
            scheduler.step_update(_iter)
            t.update(n_batch)
            
            if batch %100 ==0:
                loss, current=loss.item(), batch*n_batch
                tqdm.write(f"loss: {loss:>6f}, [{current:>5d}/{train_size:>5d}]")
    
    torch.save(model.state_dict(), os.path.join(save_path_state_dict+'state_dict_epoch_{}.pt'.format(epoch)))
    
    #Validate
    model.eval()
    loss_val=0
    with torch.no_grad():
        for batch, items in enumerate(dataloader_val):
            n_batch = items['origin'].size()[0]

            items['origin'].to(device)
            items['degraded'].to(device)
            items['interpolated'].to(device)
            
            mid_result=model(items['degraded'], items['interpolated'])
            mid_result_denorm=image_processing.denormalize(mid_result, device=device)
            loss_val = PSNR(mid_result_denorm, items['origin'])
            
            if loss_val <= min_valid_loss:
                min_valid_loss=loss_val
                min_valid_epoch = epoch
        print('validation : {}\n'.format(loss_val))

print('minimum validation {} at epoch {}'.format(min_valid_loss, min_valid_epoch))
file_rep = open(save_path_date+"/min_val_epoch.txt", "w")
file_rep.write(f'minimum validation {min_valid_loss} at epoch {min_valid_epoch}')
file_rep.close()