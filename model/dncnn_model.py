# -*- coding: utf-8 -*-
# @project : Denoise-tensorflow
# @Time : 2019-08-11 15:35 
# @Author : ZhangXiao(sinceresky@foxmail.com)
# @File : dncnn_model.py
import time
from glob import glob

import tensorflow as tf
from tensorflow.python.keras.utils import Progbar

from model.componet.train_op import configure_learning_rate, configure_optimizer
from data_util.generate_data import DataGenerator
from data_util.generate_png_file import Dataset_File
import os
import numpy as np
import cv2
from data_util.patch_util import get_patch_batch


scale = float(1 / 255.0)


def dncnn(input, is_training=True, output_channels=3):
    input = tf.cast(input, tf.float32)
    input = tf.multiply(input, scale)
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 17):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
    out = input - output
    out = tf.multiply(out, 255.0)
    if not is_training:
        out = tf.cast(out, tf.uint8)
    return out


class DnCNN_Model(object):
    def __init__(self, config, is_training=True, channel=3):
        self.config = config
        self.graph = tf.Graph()
        self.is_training = is_training
        self.optimizer_name = self.config.optimizer.lower()
        self.channel = channel

    def init_sess(self):
        if self.config.gpu >= 0:
            _gpu_options = tf.GPUOptions(
                # allow_growth=True
                per_process_gpu_memory_fraction=self.config.gpu_mem_frac
            )
            _DeviceConfig = tf.ConfigProto(device_count={"CPU": self.config.num_cpu, "GPU": 1},
                                           gpu_options=_gpu_options,
                                           intra_op_parallelism_threads=3,
                                           inter_op_parallelism_threads=3
                                           # , log_device_placement=True
                                           )
        else:
            _DeviceConfig = tf.ConfigProto(device_count={"CPU": self.config.num_cpu, "GPU": 0},
                                           intra_op_parallelism_threads=3,
                                           inter_op_parallelism_threads=3
                                           # , log_device_placement=True
                                           )
        self.sess = tf.Session(config=_DeviceConfig, graph=self.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=self.config.max_hold_save, write_version=tf.train.SaverDef.V2)

    def placeholder(self):
        self.Y = tf.placeholder(tf.float32, [None, None, None, self.channel], name='clean_image')
        self.X = tf.placeholder(tf.uint8, [None, None, None, self.channel], name='noise_image')

    def pipline(self):
        self.decoder = dncnn(self.X, is_training=self.is_training, output_channels=self.channel)

    def cal_loss(self):
        self.loss = (1.0 / self.config.batch_size) * tf.nn.l2_loss(
            tf.multiply(self.Y, scale) - tf.multiply(self.decoder, scale))

    def add_summary(self):
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self._learning_rate)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(logdir=self.config.summary_dir, flush_secs=2, graph=self.sess.graph)

    def _optimizer(self, clip=None):

        self._learning_rate = configure_learning_rate(
            learning_rate_decay_type=self.config.learning_type,
            learning_rate=self.config.learning_init,
            decay_steps=self.config.learning_decay_step,
            learning_rate_decay_rate=self.config.learning_decay_rate,
            global_step=self.global_step)

        # tf.summary.scalar('lr', self._learning_rate)

        if clip is None:
            clip_value = self.config.clip
        else:
            clip_value = clip

        with tf.variable_scope("train_step"):
            optimizer = configure_optimizer(
                optimizer_name=self.optimizer_name, learning_rate=self._learning_rate)

            if clip_value > 0:
                var_list = tf.trainable_variables()  # add on 617
                grads, var_list = zip(*optimizer.compute_gradients(self.loss, var_list))
                grads, _ = tf.clip_by_global_norm(grads, clip_value)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if clip_value > 0:  # gradient clipping if clip is positive
                    # grads, var_list = zip(*optimizer.compute_gradients(self.loss, var_list)) # ori here
                    # grads, _ = tf.clip_by_global_norm(grads, clip_value)# 防止梯度爆炸 ori here
                    self.optimizer = optimizer.apply_gradients(zip(grads, var_list), self.global_step)
                else:
                    self.optimizer = optimizer.minimize(self.loss, self.global_step)

    def load(self, checkpoint_dir):
        self.config.logger.info("======== Reading checkpoint ========")
        # saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.config.logger.info('load ckpt: {}'.format(full_path))
            self.saver.restore(self.sess, full_path)
            self.itr_num = global_step
            return True
        else:
            self.itr_num = 0
            return False

    def save(self, step):
        # saver = tf.train.Saver()
        checkpoint_dir = self.config.ckpt_dir
        self.config.logger.info("============= Saving model =============")
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.config.ckpt_name),
                        global_step=step)

    def build_train_val(self):
        self.config.logger.info('init the training val model...')
        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
            self.placeholder()
            self.pipline()
            self.cal_loss()
            self._optimizer()
            self.init_sess()
            self.add_summary()
        self.load(self.config.ckpt_dir)
        self.config.logger.info('build the training val model done!')

    def build_inference(self):
        self.config.logger.info('init the inference model...')
        with self.graph.as_default():
            # self.global_step = tf.Variable(0, trainable=False)
            self.placeholder()
            self.pipline()
            self.init_sess()
        if not self.load(self.config.ckpt_dir):
            self.config.logger.info('can not find ckpt in {}'.format(self.config.ckpt_dir))
            exit(-1)
        self.config.logger.info('build the inference model done!')

    def train(self, is_record=True):
        self.config.logger.info('the batch_size: {}'.format(self.config.batch_size))
        if not is_record:
            filepaths = glob(
                self.config.train_png_dir + 'original/*.png')
            filepaths = sorted(filepaths)  # Order the list of files
            filepaths_noisy = glob(self.config.train_png_dir + 'noisy/*.png')
            filepaths_noisy = sorted(filepaths_noisy)
            ind = list(range(len(filepaths)))
            num_batch = int(len(filepaths / self.config.batch_size))
            train_data = Dataset_File(self.sess, self.config.batch_size, filepaths, filepaths_noisy, ind)
            # train_iterator = train_data.get_batch()
        else:
            train_data = DataGenerator(self.config.tf_record_dir, 'train', self.config.batch_size, self.config.logger)
            num_batch = int(train_data.get_sample_size() / self.config.batch_size)
            # train_iterator = train_data.generate()
        start_epoch = self.itr_num // num_batch
        start_time = time.time()
        for epoch in range(start_epoch, self.config.num_epoch):
            batch_id = 0
            prog = Progbar(num_batch)
            for batch_clean, batch_noisy in train_data.generate():
                # batch_noisy = batch_noisy[np.newaxis, ...]
                # batch_clean = batch_clean[np.newaxis, ...]
                batch_clean, batch_noisy = get_patch_batch(batch_clean, batch_noisy, self.config.patch_size)
                feed = {
                    self.X: batch_noisy,
                    self.Y: batch_clean,
                }
                _, loss, _summary = self.sess.run([self.optimizer, self.loss, self.merged], feed_dict=feed)
                self.itr_num += 1
                batch_id += 1
                self.writer.add_summary(summary=_summary, global_step=self.itr_num)
                prog.update(batch_id, [('epoch', int(epoch + 1)), ('loss', loss),
                                       ('global step', self.global_step.eval(self.sess)),
                                       ('itr_num', self.itr_num),
                                       ('time', time.time() - start_time)])
                # do save
                if self.itr_num % self.config.save_itr_size == 0 and self.itr_num != 0:
                    self.save(self.itr_num)
            self.save(self.itr_num)
        self.config.logger.info('train done!')

    def predicts(self, noisy_files, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        assert len(noisy_files) != 0, 'No testing data!'
        assert self.itr_num != 0, '[!] Load weights FAILED...'
        psnr_sum = 0

        for i in range(len(noisy_files)):
            noisy = cv2.imread(noisy_files[i], 0)
            noisy = noisy.reshape((noisy.shape[0], noisy.shape[1], 1))
            ori_noisy = noisy
            # noisy = cv2.resize(noisy, (180, 180), interpolation=cv2.INTER_CUBIC)
            # noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ...]
            psnr_sum += 1
            start_time = time.time()
            output_clean_image = self.sess.run(
                [self.decoder], feed_dict={self.X: noisy})
            use_time = time.time() - start_time
            out1 = np.asarray(output_clean_image)

            out_mat = np.zeros((ori_noisy.shape[0] * 3, ori_noisy.shape[1], ori_noisy.shape[2]))
            out = out1[0, 0]
            _, bin = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            bin = bin.reshape((bin.shape[0], bin.shape[1], self.channel))
            out_mat[:ori_noisy.shape[0], :ori_noisy.shape[1]] = ori_noisy
            out_mat[ori_noisy.shape[0]: ori_noisy.shape[0] * 2, :ori_noisy.shape[1]] = out
            out_mat[ori_noisy.shape[0] * 2:, : ori_noisy.shape[1]] = bin

            cv2.imwrite(os.path.join(save_dir, os.path.basename(noisy_files[i])), out_mat)

            avg_psnr = psnr_sum / len(noisy_files)
            print("--- Test ---- Average PSNR {} time: {}---".format(avg_psnr, use_time))
        print('predict done!')
