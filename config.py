# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 15:41 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : config.py
import os
from multiprocessing import cpu_count


class ConfigDnCNN:
    def __init__(self, logger, gpu=-1):
        self.gpu = gpu
        if self.gpu >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        self.gpu_mem_frac = 0.9
        self.num_cpu = cpu_count()
        logger.info('got cpu size: {}'.format(self.num_cpu))
        # model
        self.gpu_fraction = 0.9
        self.batch_size = 1
        self.num_patches = 8  # number of patches to extract from each image
        self.patch_size = 64  # size of the patches
        self.logger = logger
        # train step
        self.learning_decay_step = 1000
        self.learning_decay_rate = 0.9
        self.learning_init = 0.001
        self.clip = 5
        self.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        self.save_itr_size = 500
        self.max_hold_save = 3
        self.save_dir = 'output/'
        self.ckpt_dir = self.save_dir + 'ckpt/'
        self.summary_dir = self.save_dir + 'summary/'
        self.optimizer = 'momentum'
        self.ckpt_name = 'denoise'
        self.num_epoch = 1000
        # dataset
        self.train_png_dir = 'dataset/train/'
        self.val_png_dir = 'dataset/test/'
        self.test_png_dir = 'dataset/test/'
        self.tf_record_dir = 'dataset/tfrecords/'
        self.init_dir()
        self.channel = 1
        # print
        logger.info('save: {}, data path: {}'.format(self.save_dir, self.train_png_dir))

    def init_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(self.summary_dir):
            self.logger.info('{} not exists, create now!'.format(self.summary_dir))
            os.mkdir(self.summary_dir)
        if not os.path.exists(self.ckpt_dir):
            self.logger.info('{} not exists, create now!'.format(self.ckpt_dir))
            os.mkdir(self.ckpt_dir)
