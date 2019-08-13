# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-12 14:35 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : patch_util.py
import numpy as np


def get_patch_batch(batch_clean, batch_noisy, patch_size, dtype=np.uint8):
    # batch_size = batch_images.shape[0]
    channel = batch_clean.shape[3]
    # x = np.zeros((batch_size, patch_size, patch_size, channel), dtype=dtype)
    # y = np.zeros((batch_size, patch_size, patch_size, channel), dtype=dtype)
    # sample_id = 0
    batch_result_clean = []
    batch_result_noisy = []
    for id in range(len(batch_clean)):
        image_clean = batch_clean[id]
        image_noisy = batch_noisy[id]
        h, w, _ = image_clean.shape
        if h >= patch_size and w >= patch_size:
            i = np.random.randint(h - patch_size + 1)
            j = np.random.randint(w - patch_size + 1)
            image_patch_clean = image_clean[i:i + patch_size, j:j + patch_size]
            image_patch_noisy = image_noisy[i:i + patch_size, j:j + patch_size]
            batch_result_clean.append(image_patch_clean.reshape((patch_size, patch_size, channel)))
            batch_result_noisy.append(image_patch_noisy.reshape((patch_size, patch_size, channel)))
    return np.asarray(batch_result_clean), np.asarray(batch_result_noisy)
