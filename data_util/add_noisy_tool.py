# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-08 12:41 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : add_noisy_tool.py

import numpy as np
import cv2
import random
import os

noise_background_dir = 'noise/'
target_w, target_h = 200, 120
# target_w_all = 200


def get_noisy_ori(mat, size):
    sig = np.linspace(0, 50, size)
    np.random.shuffle(sig)

    # image = cv2.resize(mat, (180, 180), interpolation=cv2.INTER_CUBIC)
    image = mat[:min(mat.shape[0], target_h), :min(target_w, mat.shape[1])]
    # image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_CUBIC)
    # _, bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 做一次归一化
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] != 255:
                image[y][x] = 0
            else:
                image[y][x] = 255

    # image = mat
    row, col = image.shape[0], image.shape[1]
    if len(image.shape) < 3:
        ch = 1
        image = image.reshape(row, col, ch)
    else:
        ch = image.shape[2]
    mean = 0
    if target_w != image.shape[1] or target_h != image.shape[0]:
        blank = np.full((target_h, target_w, ch), 255, dtype=np.uint8)
        y = int(abs((target_h - image.shape[0]) / 2) - 1)
        if y < 0:
            y = 0
        x = int(abs(target_w - image.shape[1]) / 2) - 1
        if x < 0:
            x = 0
        blank[y:y+image.shape[0], x:x+image.shape[1]] = image
        image = blank
    # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # image = image.reshape(image.shape[0], image.shape[1], 1)
    use_background = False
    # cv2.imshow('do', image)
    if random.randint(0, 9) < 3:
        i = random.randint(0, size - 1)
        sigma = sig[i]
        gauss = np.random.normal(mean, sigma, (target_h, target_w, ch))
        gauss = gauss.reshape(target_h, target_w, ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)
    else:
        noise_name = random.randint(1, 9)
        noise_mat = cv2.imread(os.path.join(noise_background_dir, str(noise_name) + '.png'), 0)
        noise_mat = cv2.resize(noise_mat, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)# w, h
        noise_mat = noise_mat.reshape((target_h, target_w, ch))
        for y in range(noise_mat.shape[0]):
            for x in range(noise_mat.shape[1]):
                if image[y][x] != 255:
                    noise_mat[y][x] = image[y][x]
        noisy = noise_mat
        use_background = True

    # cv2.imshow('no', noisy)
    # if noisy.shape[2] != 1:
    #     noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    # if image.shape[2] != 1:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    random_num = random.randint(0, 9)
    if random_num < 3 or (use_background and random_num < 8):
        kernel_size = (3, 3)
        img_data_blur = cv2.GaussianBlur(noisy, kernel_size, 0.8)

        cv2.normalize(img_data_blur, img_data_blur, 0, 255, cv2.NORM_MINMAX)
        noisy = np.array(img_data_blur, dtype=np.uint8)
    if random.randint(0, 9) < 8:
        shape = noisy.shape
        noisy = cv2.resize(noisy, (int(shape[1] / 2), int(shape[0] / 2)), interpolation=cv2.INTER_LANCZOS4)
        noisy = cv2.resize(noisy, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)
        noisy = noisy.reshape(shape)
    # print(noisy)
    # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         if image[y][x] < 128:
    #             image[y][x] = 0
    #         else:
    #             image[y][x] = 255
    return noisy, image
    # cv2.imwrite(os.path.join(save_dir, "noisy/%04d.png" % i), noisy)
    # cv2.imwrite(os.path.join(save_dir, "original/%04d.png" % i), image)