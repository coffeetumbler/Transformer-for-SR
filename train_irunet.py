import os, time, copy, datetime
import numpy as np
import cv2

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import json

from models.irunets import IRUNet
from utils.dataloader import get_dataloader
from utils.options import parse_args
from utils import config
from utils import image_processing
from test import test

from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler

# for multi-processing
import torch.distributed as dist
import torch.multiprocessing as mp
from threading import Timer as _timer



# Initialize the environment.
def init_process(rank, world_size, master_port, timeout=60):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(0, timeout))
    
# Timer object to stop a process
class Timer:
    def __init__(self, function=None, exception_str=''):
        self.function = function
        self.exception_str = exception_str

    def raise_exception(self, *args, **kwargs):
        if self.function != None:
            self.function(*args, **kwargs)
        raise Exception(self.exception_str)

    def start(self, time, *args, **kwargs):
        self.timer = _timer(time, self.raise_exception, args=args, kwargs=kwargs)
        self.timer.start()

    def cancel(self):
        self.timer.cancel()

    def reset(self, time, *args, **kwargs):
        temp = self.timer
        self.timer = _timer(time, self.raise_exception, args=args, kwargs=kwargs)
        self.timer.start()
        temp.cancel()


# Kill this process.
def kill_process(message=None):
    print("Current process is killed.")
    if message != None:
        print(message)
    os.kill(os.getpid(), 9)

# Synchronize all parameters of models located in all processors.
def synchronize(checkpoint_dir_file, model):
    # Read a shared checkpoint directory.
    with open(checkpoint_dir_file, 'r') as f:
        checkpoint_dir = f.readline()
    model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))
    


# SSIM configs
C1 = (0.01*1)**2
C2 = (0.03*1)**2

kernel = cv2.getGaussianKernel(11, 1.5)
window = np.outer(kernel, kernel.transpose())

# Evaluation metrics
def PSNR(A, B, device):
    diff=A-B
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
    


def main(rank, world_size, ntime, args, _model):
    # GPU setting 
    init_process(rank, world_size, args.master_port)
    cudnn.benchmark=True #input size stable -> Good
    device = torch.device('cuda:'+args.gpu_id.split(',')[rank] if torch.cuda.is_available() else 'cpu')

    model = copy.deepcopy(_model).to(device)
    
    if rank == 0:
        # Save Path
        save_path ='./logs'
        save_path_iru = save_path+'/iru'
        save_path_date = save_path_iru+'/'+ntime+'_x{}'.format(args.upscale)
        save_path_state_dict = save_path_date+'/state_dict'
        save_path_model = save_path_date+'/model'
        save_path_output = save_path_date+'/output'

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path_iru):
            os.mkdir(save_path_iru)
        if not os.path.exists(save_path_date):
            os.mkdir(save_path_date)
        if not os.path.exists(save_path_state_dict):
            os.mkdir(save_path_state_dict)
        if not os.path.exists(save_path_model):
            os.mkdir(save_path_model)
        if not os.path.exists(save_path_output):
            os.mkdir(save_path_output)

        # hyperparameters save
        with open(save_path_date+'/hyperparameters.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Log summary file
        summary_writer = SummaryWriter(save_path_date)
        torch.save(model, os.path.join(save_path_model+'/model.pt'))
    else:
        # Save path for state dict
        save_path_state_dict = './logs/iru/' + ntime + '_x{}'.format(args.upscale) + '/state_dict'

    #Loss setting
    if args.loss == 'l1':
        criterion = nn.L1Loss().to(device)
    elif args.loss == 'mse':
        criterion = nn.MSELoss().to(device)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #Dataset
    mixed_dataset = (args.train_dataset == 'mixed')
    dataloader_train = get_dataloader(lr_img_size=args.lr_img_res,
                                      batch_size=args.batch_size//world_size,
                                      setting="train",
                                      num_workers=args.num_workers,
                                      SR_mode=args.upscale,
                                      data=args.train_dataset,
                                      channel_wise_noise=args.channelwise_noise_rate,
                                      data_merge=mixed_dataset, 
                                      n_patch_in_img=args.n_patch_in_img)
    multi_epochs = args.n_patch_in_img * world_size
    train_size = len(dataloader_train.dataset) * multi_epochs

    max_iter = ((len(dataloader_train.dataset) - 1) // (args.batch_size // world_size) + 1) * (args.num_epochs // world_size)

    dataloader_val = get_dataloader(setting='test',  # or 'evaluation'
                                    data='Set5',
                                    augmentation=False,
                                    num_workers=0,
                                    lr_img_size=384//args.upscale,
                                    SR_mode=args.upscale)
    val_size = len(dataloader_val)

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_iter,
                                  lr_min=args.learning_rate*args.lr_min_rate,
                                  warmup_lr_init=args.learning_rate*args.warmup_lr_init_rate,
                                  warmup_t=int(max_iter*args.warmup_t_rate),
                                  cycle_limit=1,
                                  t_in_epochs=False)

    # For validation report, max value is best!
    max_valid_loss = 0
    max_valid_epoch = 0
    
    summary_steps = args.summary_steps // args.n_patch_in_img
    
    # Set a timer to stop process when an error occurs.
    timer = Timer(function=kill_process)
    timeout_str = 'Timeout in process {}'.format(rank)
    timer.start(120, timeout_str)

    # Training & Validation
    initial_epoch = 0 if args.start_epoch == None else args.start_epoch
    _iter = ((len(dataloader_train.dataset) - 1) // (args.batch_size // world_size) + 1) * (initial_epoch // world_size)
    scheduler.step_update(_iter)
    
    # Validate first.
    if rank == 0:
        model.eval()
        psnr_val, ssim_val = test(model, dataloader_val, 1, val_size, args.upscale, device, True)

        summary_writer.add_scalar('PSNR/Set5', psnr_val, initial_epoch)
        summary_writer.add_scalar('SSIM/Set5', ssim_val, initial_epoch)

        if psnr_val >= max_valid_loss:
            max_valid_loss = psnr_val
            max_valid_epoch = initial_epoch
        print('validation PSNR : {0}, SSIM : {1}\n'.format(psnr_val, ssim_val))
    
    for epoch in range(initial_epoch, args.num_epochs, multi_epochs):
        _epoch = epoch + multi_epochs

        #Train
        model.train()

        if rank == 0:
            with tqdm(train_size) as t:
                t.set_description('epoch : {}/{}'.format(_epoch, args.num_epochs))

                for batch, items in enumerate(dataloader_train):
                    origin = items['origin'].to(device).transpose(0, 1)
                    degraded = items['degraded'].to(device).transpose(0, 1)

                    n_batch = origin.shape[1]

                    for _origin, _degraded in zip(origin, degraded):
                        mid_result = model(_degraded)  # Interpolated inputs are removed.
                        mid_result = image_processing.denormalize(mid_result, device=device)
                        loss = criterion(mid_result, _origin) / world_size

                        optimizer.zero_grad()
                        loss.backward()

                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                        
                        # Average all gradients in workers.
                        for param in model.parameters():
                            if param.requires_grad:
                                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

                        optimizer.step()

                        _iter += 1
                        scheduler.step_update(_iter)
                        t.update(n_batch * world_size)

                    # Log summaries.
                    if (batch+1) % summary_steps == 0:
                        summary_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], _iter)
                        summary_writer.add_scalar('Training Loss', loss.item() * world_size, _iter)

                        loss, current = loss.item() * world_size, (batch+1) * n_batch * multi_epochs
                        tqdm.write(f"loss: {loss:>6f}, [{current:>5d}/{train_size:>5d}]")
                        
                    # Reset a timer.
                    timer.reset(60, timeout_str)
                        
        else:
            for items in dataloader_train:
                origin = items['origin'].to(device).transpose(0, 1)
                degraded = items['degraded'].to(device).transpose(0, 1)

                for _origin, _degraded in zip(origin, degraded):
                    mid_result = model(_degraded)  # Interpolated inputs are removed.
                    mid_result = image_processing.denormalize(mid_result, device=device)
                    loss = criterion(mid_result, _origin) / world_size

                    optimizer.zero_grad()
                    loss.backward()

                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    
                    # Average all gradients in workers.
                    for param in model.parameters():
                        if param.requires_grad:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

                    optimizer.step()

                    _iter += 1
                    scheduler.step_update(_iter)
                    
                # Reset a timer.
                timer.reset(60, timeout_str)

                
        # Validate and save the checkpoint.
        if _epoch % args.validation_steps == 0:
            # Save and synchronize model parameters.
            if rank == 0:
                torch.save(model.state_dict(), os.path.join(save_path_state_dict, 'state_dict_epoch_{}.pt'.format(_epoch)))
                dist.barrier()
            else:
                dist.barrier()
                model.load_state_dict(torch.load(os.path.join(save_path_state_dict, 'state_dict_epoch_{}.pt'.format(_epoch)), map_location='cpu'))
                
            # Validate
            if rank == 0:
                model.eval()
                psnr_val, ssim_val = test(model, dataloader_val, 1, val_size, args.upscale, device, True)
                
                summary_writer.add_scalar('PSNR/Set5', psnr_val, _epoch)
                summary_writer.add_scalar('SSIM/Set5', ssim_val, _epoch)
                
                if psnr_val >= max_valid_loss:
                    max_valid_loss = psnr_val
                    max_valid_epoch = _epoch
                print('validation PSNR : {0}, SSIM : {1}\n'.format(psnr_val, ssim_val))
                
            # Reset a timer.
            timer.reset(60, timeout_str)

    # Cancel a timer.
    timer.cancel()
    
    if rank == 0:
        print('maximum validation {} at epoch {}'.format(max_valid_loss, max_valid_epoch))
        file_rep = open(save_path_date+"/max_val_epoch.txt", "w")
        file_rep.write(f'maximum validation {max_valid_loss} at epoch {max_valid_epoch}')
        file_rep.close()
        summary_writer.close()

                                      
                                      
if __name__ == "__main__":
    args = parse_args()

    # model setting & save
    if args.checkpoint != None and args.start_epoch != None:
        model = torch.load(os.path.join(args.checkpoint, 'model/model.pt'), map_location='cpu')
        model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'state_dict/state_dict_epoch_{}.pt'.format(args.start_epoch)), map_location='cpu'))
        
        # date
        ntime = args.checkpoint[-18:-3]
    else:
        encoder_n_layer = list(map(int, args.encoder_n_layer.split(',')))
        if len(encoder_n_layer) == 1:
            encoder_n_layer = encoder_n_layer[0]
        
        if args.model == 'IRU':
            model = IRUNet(img_res=args.lr_img_res*args.upscale,
                           d_embed=args.d_embed,
                           n_layer=encoder_n_layer,
                           hidden_dim_rate=args.hidden_dim_rate,
                           dropout=args.dropout,
                           path_dropout=args.path_dropout,
                           sr_upscale=args.upscale,
                           version=args.version)
            
        if args.checkpoint != None:
            checkpoint_state_dict = torch.load(args.checkpoint, map_location='cpu')
            try:
                model.load_state_dict(checkpoint_state_dict)
            except:
                print('State dictionary does not match perfectly with model.')
                try:
                    model.load_state_dict(checkpoint_state_dict, strict=False)
                except:
                    print('Some layers do not have the same structure with the corresponding layers. Trying matching state dictionaries of encoder 1-4 only.')
                    for encoder_index in range(1, 5):
                        encoder_name = 'encoder_{}.'.format(encoder_index)
                        prefix_len = len(encoder_name)
                        sub_dict = {key[prefix_len:] : value for key, value\
                                    in checkpoint_state_dict.items() if key.startswith(encoder_name)}
                        getattr(model, encoder_name[:-1]).load_state_dict(sub_dict, strict=False)
        
        # Date
        now = time.localtime()
        ntime = "%04d%02d%02d_%02d%02d%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
                 
    # Run multi-processing.
    mp.spawn(main, args=(args.world_size, ntime, args, model), nprocs=args.world_size, join=True)
