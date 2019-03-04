import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    trainset = ['Human36M', 'MPII'] # Human36M, MPII. !!Note that 0th db is reference db!!
    testset = 'Human36M' # Human36M, MPII

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output2')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    tbx_dir = osp.join(output_dir, 'tensorboard')
 
    ## model setting
    resnet_type = 152 # 18, 34, 50, 101, 152
    
    ## input, output
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 64
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    lr_dec_epoch = [295, 298]
    end_epoch = 300
    lr = 1e-3
    lr_dec_factor = 0.1
    optimizer = 'adam'
    weight_decay = 1e-5
    batch_size = 8

    ## testing config
    test_batch_size = 4
    flip_test = False

    ## others
    num_thread = 24 #8
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

