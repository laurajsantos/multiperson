import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import mmcv
import os.path as osp
import logging
import torch.distributed as dist
import time

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
work_dir = osp.abspath('./work_dirs/tune')
mmcv.mkdir_or_exist(work_dir)
def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())
def _add_file_handler(logger,filename=None,mode='w',level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger
def init_logger(log_dir, level):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        _rank, _ = get_dist_info()
        timestamp = get_time_str()
        if log_dir and _rank == 0:
            filename = '{}.log'.format(timestamp)
            log_file = osp.join(log_dir, filename)
            _add_file_handler(logger, log_file, level=level)
        return logger
init_logger(work_dir, 'INFO')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
# from torch import nn

# import argparse
# import os.path as osp
# import sys
# import cv2
# import numpy as np
# PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# print(PROJECT_PATH)
# sys.path.insert(0, PROJECT_PATH)
# from mmcv import Config
# from mmcv.runner import Runner

# from mmcv.parallel import DataContainer as DC
# from mmcv.parallel import MMDataParallel
# from mmdet.apis.train import build_optimizer
# from mmdet.models.utils.smpl.renderer import Renderer
# from mmdet import __version__
# from mmdet.models import build_detector
# from mmdet.datasets.transforms import ImageTransform
# from mmdet.datasets.utils import to_tensor
# def parse_args():
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--config', help='train config file path')
#     parser.add_argument('--ckpt', type=str, default='')
#     args = parser.parse_args()

#     return args
# args = parse_args()
# print("Entrei")
# cfg = Config.fromfile(args.config)

# # set cudnn_benchmark
# if cfg.get('cudnn_benchmark', False):
#     torch.backends.cudnn.benchmark = True

# if args.ckpt:
#     cfg.resume_from = args.ckpt

# cfg.test_cfg.rcnn.score_thr = 0.5
# model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)