# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 15:41 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : main.py
import argparse

from init_logger import get_logger
from config import ConfigDnCNN as cfg
from model.dncnn_model import DnCNN_Model
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', type=int, default=-1, help='gpu flag, 1, 0 for GPU ID and -1 for CPU',
                    choices=(0, 1, -1))
parser.add_argument('--mode', dest='mode', type=str, default='train', help='mode for train val or test or predict',
                    choices=('train', 'predict', 'test'))

args = parser.parse_args()

if __name__ == '__main__':
    logger = get_logger('log/', 'dncnn.log', 'dncnn train')
    config = cfg(logger, args.gpu)
    is_training = args.mode == 'train'
    model = DnCNN_Model(config, is_training, channel=config.channel)
    if is_training:
        model.build_train_val()
        model.train()
    else:
        model.build_inference()
    if args.mode == 'predict':
        files = [os.path.abspath('dataset/test/' + file) for file in os.listdir('dataset/test/') if
                 file.endswith('png')]
        model.predicts(files, 'dataset/out/')
