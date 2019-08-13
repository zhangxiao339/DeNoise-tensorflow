# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 15:46 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : generate_data.py

from multiprocessing.pool import Pool

# from tensorflow.python.data import TFRecordDataset
from tqdm import tqdm

from init_logger import get_logger
from data_util.add_noisy_tool import get_noisy_ori
import os
import cv2
import tensorflow as tf
import random
import numpy as np

original_image_folder = '/Users/j.lee/Desktop/jar_out/sever_temp/zc_original_data/data_set/processed/combine_old' \
                        '/original/img_cv_out'

out_tf_records_path = './tf_records/'

max_cache_sample_size = 5000


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_example(param):
    image_folder, name, size = param
    file = os.path.join(image_folder, name)
    if not os.path.exists(file):
        return None
    mat = cv2.imread(file, 0)
    noisy, image = get_noisy_ori(mat, size)
    # cv2.imshow('noise', noisy)
    # cv2.imshow('clean', image)
    # cv2.waitKey(0)
    feature = {
        'ori': _bytes_feature(image.tobytes()),
        'ori_shape': _bytes_feature(np.asarray(image.shape, dtype=np.uint32).tobytes()),
        'noisy': _bytes_feature(noisy.tobytes()),
        'noisy_shape': _bytes_feature(np.asarray(noisy.shape, dtype=np.uint32).tobytes())
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def write_records_multi_threads(image_folder, names, tf_file, logger, num_thread):
    if names is None or len(names) < 14:
        logger.info('get image empty!')
    iter_size = int(len(names) / max_cache_sample_size)
    if iter_size * max_cache_sample_size < len(names):
        iter_size += 1
    logger.info('will spit for {} itr.'.format(iter_size))
    total_size = 0
    writer = tf.python_io.TFRecordWriter(tf_file)
    for i in range(iter_size):
        logger.info('\t {} / {} go to process...'.format(i + 1, iter_size))
        pool = Pool(num_thread)
        size = len(names)
        examples = pool.map(get_example, [(image_folder, name + '.png', size) for name in
                                          names[i * max_cache_sample_size:(i + 1) * max_cache_sample_size]])
        pool.close()
        pool.join()
        # examples = get_example((image_folder, names[0] + '.png', size))
        examples_ne = list()
        if examples is None:
            logger.info('get example none...')
            continue
        else:
            for example in examples:
                if example is None:
                    continue
                else:
                    examples_ne.append(example)
        if len(examples_ne) <= 0:
            logger.info('get example empty...')
            continue

        for example in examples_ne:
            writer.write(example.SerializeToString())
            total_size += 1
        logger.info('\t\twrite sample size: {}'.format(len(examples_ne)))
    writer.close()
    logger.info('write example size: {}'.format(total_size))


def build_tf_records_file(tf_file_dir, image_folder, logger, num_thread=14):
    if not os.path.exists(tf_file_dir):
        os.mkdir(tf_file_dir)
    logger.info('the image folder: {}'.format(image_folder))
    train_tf_file = os.path.join(tf_file_dir, 'train.tfrecords')
    val_tf_file = os.path.join(tf_file_dir,'val.tfrecords')
    logger.info('save train tf_record to: {}, val to: {}'.format(train_tf_file, val_tf_file))

    if not os.path.exists(original_image_folder):
        logger.info(original_image_folder + ' is not exists')
        exit(-1)
    else:
        logger.info('exists')
    logger.info('go to build with {} threads...'.format(num_thread))
    names = [os.path.splitext(name)[0] for name in os.listdir(original_image_folder) if name.endswith('png')]
    random.shuffle(names)
    random.shuffle(names)
    # split for train , val, test
    val_size = min(int(len(names) * 0.3), 10000)
    train_size = len(names) - val_size
    logger.info('train size: {}, val size: {}'.format(train_size, val_size))
    logger.info('go write train...')
    write_records_multi_threads(image_folder, names[:train_size + 1], train_tf_file, logger, num_thread)
    logger.info('go write val...')
    write_records_multi_threads(image_folder, names[train_size + 1:], val_tf_file, logger, num_thread)
    logger.info('all write done!')


def count_sample_size(tf_file):
    c = 0

    for _ in tf.python_io.tf_record_iterator(tf_file):
        c += 1
    return c


def parse_record(example_proto):
    features = tf.parse_single_example(example_proto,
                                       features={
                                           'ori': tf.FixedLenFeature([], tf.string),
                                           'ori_shape': tf.FixedLenFeature([], tf.string),
                                           'noisy': tf.FixedLenFeature([], tf.string),
                                           'noisy_shape': tf.FixedLenFeature([], tf.string)
                                       }
                                       )
    return features['ori'], features['ori_shape'], features['noisy'], features['noisy_shape']


def generate_for_dncnn_train_data(mat):
    mat = mat.astype('float32') / 255.0
    mat = mat[np.newaxis, ...]
    return mat


class DataGenerator:
    def __init__(self, tf_records_dir, type, batch_size, logger):
        self.tf_records_dir = tf_records_dir
        self.type = type
        self.batch_size = batch_size
        self.tf_file = os.path.abspath(os.path.join(tf_records_dir, type + '.tfrecords'))
        assert os.path.exists(self.tf_file), 'the tf_records_file not exists, the file: {}'.format(self.tf_file)
        self.logger = logger
        self.sample_size = count_sample_size(self.tf_file)
        self.logger.info('got the sample size: {}'.format(self.sample_size))
        self.database = tf.data.TFRecordDataset([self.tf_file])
        self.database = self.database.map(parse_record)
        self.database.shuffle(1)

    def get_sample_size(self):
        return self.sample_size

    def generate(self):
        iterator = self.database.batch(self.batch_size).make_one_shot_iterator()
        original, original_sp, noisy, noisy_sp = iterator.get_next()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            while True:
                try:
                    batch_ori = []
                    batch_noisy = []
                    ori_images, ori_shapes, noisy_images, noisy_shapes = sess.run([original, original_sp, noisy, noisy_sp])
                    for ori_img,ori_shape_bytes, noisy_img, noisy_shape_bytes in zip(ori_images, ori_shapes, noisy_images, noisy_shapes):
                        ori_mat = np.frombuffer(ori_img, dtype=np.uint8)
                        ori_shape = np.frombuffer(ori_shape_bytes, dtype=np.uint32)
                        noisy_mat = np.frombuffer(noisy_img, dtype=np.uint8)
                        noisy_shape = np.frombuffer(noisy_shape_bytes, dtype=np.uint32)
                        if len(ori_shape) == 2:
                            ori_mat = ori_mat.reshape((ori_shape[0], ori_shape[1], 1))
                        else:
                            ori_mat = ori_mat.reshape(ori_shape)
                        if len(noisy_shape) == 2:
                            noisy_mat = noisy_mat.reshape((noisy_shape[0], noisy_shape[1], 1))
                        else:
                            noisy_mat = noisy_mat.reshape(noisy_shape)
                        # cv2.imshow('ori', ori_mat)
                        # cv2.imshow('noise', noisy_mat)
                        # cv2.waitKey(0)
                        # noisy_mat = generate_for_dncnn_train_data(noisy_mat)
                        # ori_mat = generate_for_dncnn_train_data(ori_mat)
                        batch_noisy.append(noisy_mat)
                        batch_ori.append(ori_mat)
                    yield np.asarray(batch_ori), np.asarray(batch_noisy)
                except tf.errors.OutOfRangeError:
                    self.logger.info('read tf records out of range!')
                    break

    def generate_sess(self, sess):
        iterator = self.database.batch(self.batch_size).make_one_shot_iterator()
        original, original_sp, noisy, noisy_sp = iterator.get_next()
        while True:
            try:
                batch_ori = []
                batch_noisy = []
                ori_images, ori_shapes, noisy_images, noisy_shapes = sess.run([original, original_sp, noisy, noisy_sp])
                for ori_img, ori_shape_bytes, noisy_img, noisy_shape_bytes in zip(ori_images, ori_shapes, noisy_images,
                                                                                  noisy_shapes):
                    ori_mat = np.frombuffer(ori_img, dtype=np.uint8)
                    ori_shape = np.frombuffer(ori_shape_bytes, dtype=np.uint32)
                    noisy_mat = np.frombuffer(noisy_img, dtype=np.uint8)
                    noisy_shape = np.frombuffer(noisy_shape_bytes, dtype=np.uint32)
                    if len(ori_shape) == 2:
                        ori_mat = ori_mat.reshape((ori_shape[0], ori_shape[1], 1))
                    else:
                        ori_mat = ori_mat.reshape(ori_shape)
                    if len(noisy_shape) == 2:
                        noisy_mat = noisy_mat.reshape((noisy_shape[0], noisy_shape[1], 1))
                    else:
                        noisy_mat = noisy_mat.reshape(noisy_shape)
                    # cv2.imshow('ori', ori_mat)
                    # cv2.imshow('noise', noisy_mat)
                    # cv2.waitKey(0)
                    # noisy_mat = generate_for_dncnn_train_data(noisy_mat)
                    # ori_mat = generate_for_dncnn_train_data(ori_mat)
                    batch_noisy.append(noisy_mat)
                    batch_ori.append(ori_mat)
                yield np.asarray(batch_ori), np.asarray(batch_noisy)
            except tf.errors.OutOfRangeError:
                self.logger.info('read tf records out of range!')
                break


if __name__ == '__main__':
    logger = get_logger('./log/', 'generate_data.log', 'generate')
    logger.info('logger start....')
    build_tf_records_file(tf_file_dir='./tf_records/', image_folder=original_image_folder, logger=logger,
                          num_thread=14)
    # for i in range(1, 9):
    #     mat = cv2.imread(os.path.join('noise/', str(i) + '.png'), 0)
    #     mat = cv2.resize(mat, (200, 120), interpolation=cv2.INTER_LANCZOS4)
    #     cv2.imshow('nosie', mat)
    #     mat = mat.reshape((120, 200, 1))
    #     cv2.imshow('reshape', mat)
    #     cv2.waitKey(0)

    # dataset = DataGenerator('../dataset/tfrecords/', 'train', 1, logger)
    # for clean_batch, noisy_batch in dataset.generate():
    #     for i in range(len(clean_batch)):
    #         clean = clean_batch[i]
    #         noisy = noisy_batch[i]
    #         cv2.imshow('clean', clean)
    #         cv2.imshow('noise', noisy)
    #         cv2.waitKey(0)
    # logger.info('get sample size: {}'.format(dataset.get_sample_size()))
    # for batch_image, batch_noisy in dataset.generate():
    #     print(type(batch_image), type(batch_noisy))

    '''save to png file start'''
    # id = 0
    # for batch_ori, batch_noisy in tqdm(dataset.generate()):
    #     # print(batch_ori.shape, batch_noisy.shape)
    #     for i in range(len(batch_ori)):
    #         file = os.path.abspath(os.path.join('../data/train/original/', str(id) + '.png'))
    #         cv2.imwrite(file, batch_ori[i])
    #         file = os.path.abspath(os.path.join('../data/train/noisy/', str(id) + '.png'))
    #         cv2.imwrite(file, batch_noisy[i])
    #         id += 1
    # print('write train file size: {}'.format(id))
    # dataset = DataGenerator('../dataset/tfrecords/', 'val', 1, logger)
    # logger.info('get sample size: {}'.format(dataset.get_sample_size()))
    # id = 0
    # for batch_ori, batch_noisy in tqdm(dataset.generate()):
    #     # print(batch_ori.shape, batch_noisy.shape)
    #     for i in range(len(batch_ori)):
    #         file = os.path.abspath(os.path.join('../data/test/original/', str(id) + '.png'))
    #         # print(file)
    #         cv2.imwrite(file, batch_ori[i])
    #         file = os.path.abspath(os.path.join('../data/test/noisy/', str(id) + '.png'))
    #         # print(file)
    #         cv2.imwrite(file, batch_noisy[i])
    #         id += 1
    # print('write test file size: {}'.format(id))
    '''save png to file end'''

    # if not os.path.exists(original_image_folder):
    #     logger.info(original_image_folder + ' is not exists')
    #     exit(-1)
    # else:
    #     logger.info('exists')
    # names = [os.path.splitext(name)[0] for name in os.listdir(original_image_folder)
    #          if name.endswith('png')]
    # print(len(names))
    #
    # for name in names:
    #     mat = cv2.imread(os.path.join(original_image_folder, name + '.png'), 0)
    #     print('in shape: ', mat.shape)
    #     cv2.imshow('in', mat)
    #
    #     noisy, image = get_noisy_ori(mat, len(names))
    #     cv2.imshow('do', image)
    #     print(noisy.shape)
    #     cv2.imshow('noisy', noisy)
    #     cv2.waitKey(0)
    logger.info('all done!')