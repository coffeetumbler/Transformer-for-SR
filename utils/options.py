import argparse

"""
train.py 파일 안에서 아래처럼 사용하면 됩니다.
필요한 options은 여기에 각자 추가해주세요.

from utils.options import parse_args
args = parse_args()
"""

def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--', type=, default=,
#                         help='')
    
    # General options
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--master_port', default='12355', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--start_epoch', default=None, type=int)
    
    # Model options
    parser.add_argument('--model', type=str, default='IRT')
    parser.add_argument('--lr_img_res', type=int, default=48,
                        help='Resolution of low-resolution image')
    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--intermediate_upscale', default=False, action='store_true',
                        help='x2 upscaling twice for x4 upscaling')
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--d_embed', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--encoder_n_layer', type=str, default='12',
                        help='Number of encoder layers; total number of layers or list of each number of layer (with seperater ",")')
    parser.add_argument('--decoder_n_layer', type=int, default=12,
                        help='Number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--interpolated_decoder_input', default=False, action='store_true',
                        help='Interpolated images for decoder inputs')
    parser.add_argument('--decoder_input_from_encoder', dest='raw_decoder_input', default=True,
                        action='store_false', help='Make decoder inputs from encoder outputs')
    parser.add_argument('--n_residual_block', type=int, default=3)
    parser.add_argument('--hidden_dim_rate', type=int, default=2)
    parser.add_argument('--version', type=float, default=1.1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--path_dropout', type=float, default=0.1)
    
    # Training options
    parser.add_argument('--train_dataset', type=str, default='mixed',
                        help='Training dataset; mixed/DIV2K/BSDS200/General100')
    parser.add_argument('--test_dataset', type=str, default='BSDS100',
                        help='Test dataset; DIV2K/BSDS100/Urban100/Manga109/Set5/Set14')
    parser.add_argument('--valid_dataset', type=str, default='DIV2K',
                        help='Validation dataset; DIV2K/BSDS100/Urban100/Manga109/Set5/Set14')
    parser.add_argument('--n_patch_in_img', type=int, default=4,
                        help='Number of patches in an original image per epoch')
    parser.add_argument('--channelwise_noise_rate', type=float, default=0.5)
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--loss', type=str, default='l1',
                        help='Loss type; l1/mse')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_min_rate', type=float, default=0.01)
    parser.add_argument('--warmup_lr_init_rate', type=float, default=0.001)
    parser.add_argument('--warmup_t_rate', type=float, default=0.1)
    
#     고정해도 됩니다.
#     parser.add_argument('--t_in_epochs', type=bool, default=False)
#     parser.add_argument('--cycle_limit', type=int, default=1)

    parser.add_argument('--grad_clip_norm', default=1, type=float,
                        help='Maximum gradient norm in gradient clipping')
    parser.add_argument('--summary_steps', default=40, type=int,
                        help='Training summary frequency')
    parser.add_argument('--validation_steps', default=10, type=int,
                        help='Validation and checkpoint saving frequency')
    
    return parser.parse_args()
