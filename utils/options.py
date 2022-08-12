import argparse

"""
train.py 파일 안에서 아래처럼 사용하면 됩니다.
필요한 options은 여기에 각자 추가해주세요.

from utils.options import parse_args
args = parse_args()
"""

def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--', type=, defualt=,
#                         help='')
    
    # General options
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # Model options
    parser.add_argument('--lr_img_res', type=int, defualt=48,
                        help='Resolution of low-resolution image')
    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--intermediate_upscale', defualt=False, action='store_true',
                        help='x2 upscaling twice for x4 upscaling')
    parser.add_argument('--patch_size', type=int, defualt=2)
    parser.add_argument('--window_size', type=int, defualt=4)
    parser.add_argument('--d_embed', type=int, defualt=128,
                        help='Embedding dimension')
    parser.add_argument('--encoder_n_layer', type=int, defualt=12,
                        help='Number of encoder layers')
    parser.add_argument('--decoder_n_layer', type=int, defualt=12,
                        help='Number of decoder layers')
    parser.add_argument('--n_head', type=int, defualt=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, defualt=0.1)
    
    # Training options
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', '--learning_rate', type=float, defualt=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--loss', type=str, default='l1',
                        help='Loss type; l1/mse')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr_min_rate', type=float, default=0.01)
    parser.add_argument('--warmup_lr_init_rate', type=float, default=0.001)
    parser.add_argument('--warmup_t_rate', type=float, default=0.1)
    
#     고정해도 됩니다.
#     parser.add_argument('--t_in_epochs', type=bool, default=False)
#     parser.add_argument('--cycle_limit', type=int, default=1)

    parser.add_argument('--grad_clip_norm', default=0.1, type=float,
                        help='Maximum gradient norm in gradient clipping')
    parser.add_argument('--summary_steps', default=300, type=int,
                        help='Training summary frequency')
    
    return parser.parse_args()