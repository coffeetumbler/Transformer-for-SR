import os, sys
import time
import numpy as np
import cv2
import torch
from torch import nn
import torch.backends.cudnn as cudnn

import json
from models import whole_models

from utils.dataloader import get_dataloader, valid_batch_making
from utils.options_test import parse_args  # option arguments
from utils.functions import simple_upscale
from utils import config
from utils import image_processing



# SSIM configs
C1 = (0.01*255)**2
C2 = (0.03*255)**2

kernel = cv2.getGaussianKernel(11, 1.5)
window = np.outer(kernel, kernel.transpose())

# Evaluation metrics
def PSNR(A, B, trim_size, device):
    diff = A - B
    diff = diff[:, :, trim_size:-trim_size, trim_size:-trim_size]
    diff = image_processing.convert_y_channel(diff, add_bias=False, device=device)
    
    mse = (diff**2).mean(dim=(1,2))
    return (-10)*torch.log10(mse).sum()


def cal_SSIM(A, B):
    '''[5:-5, 5:-5] could be skipped'''
    mu1 = cv2.filter2D(A, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(B, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(A**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(B**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(A*B, -1, window)[5:-5, 5:-5] -mu1_mu2

    ssim = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim.mean()

def SSIM(A, B, trim_size, device, test_y_channel=True):
    if trim_size > 0:
        A = A[:, :, trim_size:-trim_size, trim_size:-trim_size]
        B = B[:, :, trim_size:-trim_size, trim_size:-trim_size]
    
    if test_y_channel == True:
        A = image_processing.convert_y_channel(A, add_bias=True, device=device)
        B = image_processing.convert_y_channel(B, add_bias=True, device=device)
        
        A = A.cpu().numpy()
        B = B.cpu().numpy()
        
        ssims = []
        for _A, _B in zip(A, B):
            ssims.append(cal_SSIM(_A, _B))
        return np.sum(ssims)

    else:
        A = A.cpu().numpy().transpose(0, 2, 3, 1) * 255
        B = B.cpu().numpy().transpose(0, 2, 3, 1) * 255
        
        ssims = []
        for _A, _B in zip(A, B):
            ssims.append(cal_SSIM(_A, _B))
        return np.sum(ssims)



def test(model, dataloader_test, batch_size, test_size, trim_size, device,
         pad_boundary=False, post_process=False, save_memory=False, save_image=None):
    psnr_test = []
    ssim_test = []

    with torch.no_grad():
        for th, items in enumerate(dataloader_test):
            # Patched inputs
            if dataloader_test.dataset.setting_index == 2:
                origin = items["origin"].to(device)  # (1, 3, H, W)
                img = torch.zeros(origin.shape).to(device) # origin.shape[0], origin.shape[1] # (1, 3, H, W)

                batch_making = valid_batch_making(items=items, batch=batch_size)

                for keys, degraded_items in batch_making:
                    degraded_items= degraded_items.to(device)
                    degradeds = model.evaluate(degraded_items)

                    img_h, img_w = degradeds[0].shape[1], degradeds[0].shape[2]

                    for i, key in enumerate(keys):
                        p, q = key[0], key[1]
                        img[..., p:p+img_h, q:q+img_w] += degradeds[i] 

                mask = items["mask"].to(device)    
                img /= mask
            
            # Whole inputs
            elif dataloader_test.dataset.setting_index == 3:
                origin = items["origin"].to(device)  # (1, 3, H, W)
                degraded = items["degraded"].to(device)
                
                img = model.evaluate(degraded, pad_boundary, post_process, save_memory)
            
            img = image_processing.denormalize(img, device=device)
            img = image_processing.quantize(img).to(torch.float64)
            origin = origin.to(torch.float64)

            psnr = PSNR(img, origin, trim_size, device).item()
            ssim = SSIM(img, origin, trim_size, device)

            if save_image != None:
                print('PSNR : {}, SSIM : {} for {}-th image'.format(psnr, ssim, th+1))
#                 cv2.imwrite(save_image+'/final_{0}.png'.format(th), img[0].permute(1,2,0).cpu().numpy()*255)

            psnr_test.append(psnr)
            ssim_test.append(ssim)
            
        psnr_test = np.sum(psnr_test) / test_size
        ssim_test = np.sum(ssim_test) / test_size

    return psnr_test, ssim_test
     
    
    
if __name__ == '__main__':
    args = parse_args() 

    #Date
    now=time.localtime()
    ntime="%04d%02d%02d_%02d%02d%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    #Save Path
    save_path ='./test_result'
    save_path_upscale = save_path+'/x'+str(args.upscale)
    save_path_date = save_path_upscale+'/'+ntime
    save_path_output = save_path_date+'/output'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_upscale):
        os.mkdir(save_path_upscale)
    if not os.path.exists(save_path_date):
        os.mkdir(save_path_date)
    if not os.path.exists(save_path_output):
        os.mkdir(save_path_output)

    #hyperparameters save
    with open(save_path_date+'/hyperparameters.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    #GPU setting
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    #Data Load
    if args.lr_img_res == None:
        dataloader_test = get_dataloader(setting='test', augmentation=False, num_workers=0,
                                         SR_mode=args.upscale, data=args.test_dataset)
    else:
        dataloader_test = get_dataloader(setting='evaluation', augmentation=False, num_workers=0,
                                         SR_mode=args.upscale, data=args.test_dataset, lr_img_size=args.lr_img_res)
    test_size = len(dataloader_test)

    #Load Model & Weight
#     model_path = args.logs_path+'/x'+str(args.upscale)+'/'+args.from_ntime+'/model/model.pt'
    model_path = args.logs_path+'/x'+str(args.upscale)+'/'+args.model_path
    model = torch.load(model_path, map_location=device)
    
#     weight_path = args.logs_path+'/x'+str(args.upscale)+'/'+args.from_ntime+'/state_dict/state_dict_epoch_'+str(args.from_epochnum)+'.pt'
    weight_path = args.logs_path+'/x'+str(args.upscale)+'/'+args.state_dict_path
    weight = torch.load(weight_path, map_location='cpu')
    weight_keys = list(weight.keys())
    
    for key in weight_keys:
        if '.mask' in key:
            del weight[key]
            
    model.load_state_dict(weight, strict=False)
    model.eval()
    
    psnr_test, ssim_test = test(model, dataloader_test, args.batch_size, test_size, args.upscale, device,
                                args.pad_boundary, args.trim_boundary, args.save_memory,
                                save_path_output if args.output_results else None)

    print(f'average PSNR : {psnr_test}, average SSIM : {ssim_test}')
    
    file_rep = open(save_path_output+"/test_result.txt", "w")
    file_rep.write(f'PSNR : {psnr_test} \nSSIM : {ssim_test}')
    file_rep.close()