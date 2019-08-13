# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 21:40 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : generate_png_file.py
import tensorflow as tf
import time
import random


class Dataset_File(object):
    def __init__(self, sess, batch_size, file_paths, file_noisy_paths, ind):
        self.sess = sess
        seed = time.time()
        random.seed(seed)

        random.shuffle(ind)

        filenames = list()
        for i in range(len(file_paths)):
            filenames.append(file_noisy_paths[ind[i]])
            filenames.append(file_paths[ind[i]])

        # Parameters
        num_patches = 8  # number of patches to extract from each image
        patch_size = 64  # size of the patches
        num_parallel_calls = 1  # number of threads
        self.batch_size = batch_size  # size of the batch
        get_patches_fn = lambda image: get_patches(image, num_patches=num_patches, patch_size=patch_size)
        dataset = (
            tf.data.Dataset.from_tensor_slices(filenames)
                .map(im_read, num_parallel_calls=num_parallel_calls)
                .map(get_patches_fn, num_parallel_calls=num_parallel_calls)
                .batch(self.batch_size)
                .prefetch(self.batch_size)
        )

        iterator = dataset.make_one_shot_iterator()
        self.iter = iterator.get_next()

    def generate(self):
        res = self.sess.run(self.iter)
        return res


def im_read(filename):
    """Decode the png image from the filename and convert to [0, 1]."""
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image


def get_patches(image, num_patches=128, patch_size=64):
    """Get `num_patches` from the image"""
    patches = []
    for i in range(num_patches):
        point1 = random.randint(0, 116)  # 116 comes from the image source size (180) - the patch dimension (64)
        point2 = random.randint(0, 116)
        patch = tf.image.crop_to_bounding_box(image, point1, point2, patch_size, patch_size)
        patches.append(patch)
    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 3]
    return patches
