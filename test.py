from __future__ import absolute_import, division, print_function

import os
import utils
import models
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im

from glob import glob


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--lambda_', dest='lambda_', type=float, default=10.0, help='lambda')
parser.add_argument('--channel', dest='channel', type=int, default=3, help='image channel')
parser.add_argument('--ratio', dest='ratio', type=int, default=1, help='width/height ratio')
args = parser.parse_args()

dataset = args.dataset
crop_size = args.crop_size
channel=args.channel
lambda_ = args.lambda_
ratio = args.ratio

""" run """
with tf.Session() as sess:
    # nodes
    x = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size*ratio, channel])
    R_x_history = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size*ratio, channel])

    R_x = models.refiner(x, 'R_x')

    # retore
    saver = tf.train.Saver()
    ckpt_path = utils.load_checkpoint('./checkpoints/' + dataset + "_" + str(lambda_), sess, saver)
    if ckpt_path is None:
        print (ckpt_path)
        raise Exception('No checkpoint!')
    else:
        print('Copy variables from % s' % ckpt_path)

    # test
    x_list = glob('./datasets/' + dataset + '/vkitti_1.3.1_rgb/0018/morning/*.png')

    x_save_dir = './test_predictions/' + dataset + '/vkitti_1.3.1_rgb_refinedv2/0018/morning/'
    utils.mkdir([x_save_dir])
    for i in range(len(x_list)):
        x_real_ipt = im.imresize(im.imread(x_list[i],mode='RGB'), [crop_size, crop_size*ratio])
        x_real_ipt.shape = 1, crop_size, crop_size*ratio, 3
        R_x_opt = sess.run(R_x, feed_dict={x: x_real_ipt})
        sample_opt = R_x_opt
        img_name = os.path.basename(x_list[i])
        print (x_list[i])
        im.imwrite(im.immerge(sample_opt, 1, 1), x_save_dir + img_name)
        print('Save %s' % (x_save_dir + img_name))

