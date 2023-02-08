import argparse

def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--', type=, default=,
#                         help='')
    
    # General options
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # Model options
    parser.add_argument('--lr_img_res', type=int, default=None,
                        help='Resolution of low-resolution image, must be a multiple of 24; None for maximum size automatically')
    parser.add_argument('--upscale', type=int, default=2)

    #Load options
    parser.add_argument('--logs_path', type=str, default='./logs')
#     parser.add_argument('--from_ntime', type=str, default='20220830_200055')
#     parser.add_argument('--from_epochnum', type=int, default = 10000)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--state_dict_path', type=str, default='')

    # Test options
    parser.add_argument('--test_dataset', type=str, default='Set5',
                        help='Test dataset; DIV2K/BSDS100/Urban100/Manga109/Set5/Set14')
    parser.add_argument('--output_results', default=False, action='store_true',
                        help='Output image saving option')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--trim_boundary', default=False, action='store_true',
                        help='Trim inner boundary of image patches')
    parser.add_argument('--pad_boundary', default=False, action='store_true',
                        help='Pad image boundaries')
    parser.add_argument('--save_memory', default=False, action='store_true',
                        help='Save memory when computing attentions')
                        
    return parser.parse_args()