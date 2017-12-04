from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ops
import data
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
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--channel', dest='channel', type=int, default=3, help='image channel')
parser.add_argument('--lambda_', dest='lambda_', type=float, default=10.0, help='lambda')
parser.add_argument('--ratio', dest='ratio', type=int, default=1, help='width/height ratio')
args = parser.parse_args()

dataset = args.dataset
load_size = args.load_size
crop_size = args.crop_size
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
gpu_id = args.gpu_id
channel=args.channel
lambda_ = args.lambda_
ratio = args.ratio


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' graph '''
    # nodes
    x = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size*ratio, channel])
    y = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size*ratio, channel])

    R_x_history = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size*ratio, channel])

    R_x = models.refiner_cyc(x, 'R_x')

    D_y = models.discriminator_global(y, 'd')
    D_R_x = models.discriminator_global(R_x, 'd', reuse=True)
    D_R_x_history = models.discriminator_global(R_x_history, 'd', reuse=True)

    # losses
    realism_loss = tf.identity(ops.l2_loss(D_R_x, tf.ones_like(D_R_x)), name='realism_loss')
    regularization_loss = tf.identity(ops.l1_loss(R_x, x) * lambda_, name='regularization_loss')
    generator_loss = tf.identity((realism_loss + regularization_loss)/2.0, name="generator_loss")

    refiner_d_loss = tf.identity(ops.l2_loss(D_R_x, tf.zeros_like(D_R_x)), name='refiner_d_loss')
    real_d_loss = tf.identity(ops.l2_loss(D_y, tf.ones_like(D_y)), name='real_d_loss')
    discrim_loss = tf.identity((refiner_d_loss + real_d_loss)/2.0, name="discriminator_loss")

    # with history
    refiner_d_loss_with_history = tf.identity(ops.l2_loss(D_R_x_history, tf.zeros_like(D_R_x_history)), 
        name='refiner_d_loss_with_history')

    discrim_loss_with_history = tf.identity((refiner_d_loss_with_history + real_d_loss) / 2.0, 
        name="discrim_loss_with_history")

    # summaries
    refiner_summary = ops.summary_tensors([realism_loss, regularization_loss])
    refiner_summary_all = ops.summary(generator_loss)

    discrim_summary = ops.summary_tensors([refiner_d_loss, real_d_loss])
    discrim_summary_all = ops.summary(discrim_loss)

    discrim_summary_with_history = ops.summary_tensors([refiner_d_loss_with_history, real_d_loss])
    discrim_summary_with_history_all = ops.summary(discrim_loss_with_history)


    ''' optim '''
    t_var = tf.trainable_variables()
    d_a_var = [var for var in t_var if 'd_discriminator' in var.name]
    g_var = [var for var in t_var if 'R_x_generator' in var.name]

    d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(discrim_loss_with_history, var_list=d_a_var)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(generator_loss, var_list=g_var)


""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = ops.counter()

'''data'''
x_img_paths = glob('./datasets/' + dataset + '/trainA/*.png')
y_img_paths = glob('./datasets/' + dataset + '/trainB/*.png')
x_data_pool = data.ImageData(sess, x_img_paths, batch_size, channels=channel, load_size=load_size, crop_size=crop_size, ratio=ratio)
y_data_pool = data.ImageData(sess, y_img_paths, batch_size, channels=channel, load_size=load_size, crop_size=crop_size, ratio=ratio)

x_test_img_paths = glob('./datasets/' + dataset + '/trainA/*.png')
y_test_img_paths = glob('./datasets/' + dataset + '/trainB/*.png')
x_test_pool = data.ImageData(sess, x_test_img_paths, batch_size, channels=channel, load_size=load_size, crop_size=crop_size, ratio=ratio)
y_test_pool = data.ImageData(sess, y_test_img_paths, batch_size, channels=channel, load_size=load_size, crop_size=crop_size, ratio=ratio)

R_x_pool = utils.ItemPool()

'''summary'''
summary_writer = tf.summary.FileWriter('./summaries/' + dataset + "_" + str(lambda_), sess.graph)

'''saver'''
ckpt_dir = './checkpoints/' + dataset + "_" + str(lambda_)
utils.mkdir(ckpt_dir + '/')

saver = tf.train.Saver(max_to_keep=5)
ckpt_path = utils.load_checkpoint(ckpt_dir, sess, saver)
if ckpt_path is None:
    sess.run(tf.global_variables_initializer())
    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print("Pretrain refiner")
        for it in range(200): 
            x_real_ipt = x_data_pool.batch()
            refiner_summary_opt, _ = sess.run([refiner_summary, g_train_op], feed_dict={x: x_real_ipt})
            summary_writer.add_summary(refiner_summary_opt, it)
        save_path = saver.save(sess, '%s/pretrained_refiner.ckpt' % (ckpt_dir))

        print("Pretrain descriminator")
        for it in range(50):            
            # prepare data
            x_real_ipt = x_data_pool.batch()
            y_real_ipt = y_data_pool.batch()
            R_x_opt = sess.run(R_x, feed_dict={x: x_real_ipt})
            R_x_sample_ipt = np.array(R_x_pool(list(R_x_opt)))

            # train D_a
            discrim_summary_opt, _ = sess.run([discrim_summary_with_history, d_a_train_op], feed_dict={y: y_real_ipt, R_x_history: R_x_sample_ipt})
            summary_writer.add_summary(discrim_summary_opt, it)
	print("Finish pretrain")
        save_path = saver.save(sess, '%s/pretrained_discriminator.ckpt' % (ckpt_dir))       

    except Exception, e:
        coord.request_stop(e)
	print (e)
    #finally:
    #    print("Stop threads and close session!")
    #    coord.request_stop()
    #    coord.join(threads)
    #    sess.close()       
else:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('Copy variables from % s' % ckpt_path)

'''train'''
try:
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_epoch = min(len(x_data_pool), len(y_data_pool)) // batch_size
    max_it = epoch * batch_epoch

    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # prepare data
        x_real_ipt = x_data_pool.batch()
        y_real_ipt = y_data_pool.batch()
        R_x_opt = sess.run(R_x, feed_dict={x: x_real_ipt})
        R_x_sample_ipt = np.array(R_x_pool(list(R_x_opt)))

        # train G
        for k in xrange(2):
            refiner_summary_opt, _ = sess.run([refiner_summary, g_train_op], feed_dict={x: x_real_ipt})
            summary_writer.add_summary(refiner_summary_opt, it*2+k)

        # train D_a
        for k in xrange(1):
            discrim_summary_opt, _ = sess.run([discrim_summary_with_history, d_a_train_op], feed_dict={y: y_real_ipt, R_x_history: R_x_sample_ipt})
            summary_writer.add_summary(discrim_summary_opt, it)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # display
        if it % 10 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 100 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            x_real_ipt = x_test_pool.batch()
            R_x_opt = sess.run(R_x, feed_dict={x: x_real_ipt})
            sample_opt = np.concatenate((x_real_ipt[0:2], R_x_opt[0:2]), axis=0)
	    print (sample_opt.shape)            
            save_dir = './sample_images_while_training/' + dataset + "_" + str(lambda_)
            utils.mkdir(save_dir + '/')
            im.imwrite(im.immerge(sample_opt, 2, 2), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception, e:
    coord.request_stop(e)
finally:
    print("Stop threads and close session!")
    coord.request_stop()
    coord.join(threads)
    sess.close()
