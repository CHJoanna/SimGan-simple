from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim


conv = functools.partial(slim.conv2d, activation_fn=None)
relu = tf.nn.relu
lrelu = functools.partial(ops.leak_relu, leak=0.2)

max_pool2d = functools.partial(slim.max_pool2d)

def discriminator(img, scope, df_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)
    print ("disciminator input", img)
    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        h0 = lrelu(conv(img, 96, 3, 2, scope='h0_conv', padding='SAME'))    # h0 is (128 x 128 x df_dim)
        h1 = lrelu(bn(conv(h0, 64, 3, 2, scope='h1_conv', padding='SAME'), scope='h1_bn'))  # h1 is (64 x 64 x df_dim*2)
        m1 = max_pool2d(h1, 3, 1, scope="max_1")
        h2 = lrelu(bn(conv(m1, 32, 3, 1, scope='h2_conv', padding='SAME'), scope='h2_bn'))  # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(bn(conv(h2, 32, 3, 1, scope='h3_conv', padding='SAME'), scope='h3_bn'))  # h3 is (32 x 32 x df_dim*8)
        logits = conv(h3, 2, 1, 1, scope='h4_conv', padding='SAME')  # h4 is (32 x 32 x 1)
        print ("disciminator output",logits)
        return logits

def refiner(img, scope, gf_dim=64, reuse=False, train=True):
    print ("refiner input",img)
    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    def residule_block(x, dim, scope='res'):
        # y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(conv(x, dim, 3, 1, padding='SAME', scope=scope + '_conv1'), scope=scope + '_bn1'))
        # y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(conv(y, dim, 3, 1, padding='SAME', scope=scope + '_conv2'), scope=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        # c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(conv(img, gf_dim, 3, 1, padding='SAME', scope='c1_conv'), scope='c1_bn'))


        r1 = residule_block(c1, gf_dim * 1, scope='r1')
        r2 = residule_block(r1, gf_dim * 1, scope='r2')
        r3 = residule_block(r2, gf_dim * 1, scope='r3')
        r4 = residule_block(r3, gf_dim * 1, scope='r4')
        r5 = residule_block(r4, gf_dim * 1, scope='r5')
        r6 = residule_block(r5, gf_dim * 1, scope='r6')
        r7 = residule_block(r6, gf_dim * 1, scope='r7')
        r8 = residule_block(r7, gf_dim * 1, scope='r8')
        r9 = residule_block(r8, gf_dim * 1, scope='r9') 
        pred = conv(r9, 3, 1, 1, padding='SAME', scope='pred_conv')
        print ("refiner output",pred)
        pred = tf.nn.tanh(pred)
        return pred





