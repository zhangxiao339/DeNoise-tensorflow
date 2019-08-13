# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 21:47 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : train_uitl.py
import numpy as np


def cal_psnr(im1, im2):  # PSNR function for 0-255 values
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def psnr_scaled(im1, im2):  # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr
