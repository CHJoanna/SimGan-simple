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


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)
def bn(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def fully_connected(input_tensor, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return activation_fn(logits)

def discriminator(img, scope, df_dim=64, reuse=False, train=True):

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
    print ("refiner input - no deconv",img)

    def residule_block(x, dim, scope='res'):
        # y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(conv(x, dim, 3, 1, padding='SAME', scope=scope + '_conv1'), name=scope + '_bn1'))
        # y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(conv(y, dim, 3, 1, padding='SAME', scope=scope + '_conv2'), name=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        # c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(conv(img, gf_dim, 3, 1, padding='SAME', scope='c1_conv'), name='c1_bn'))


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

def discriminator_cyc(img, scope, df_dim=64, reuse=False, train=True):

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
	h0 = lrelu(conv2d(img, df_dim, name='d_h0_conv'))
        h1 = lrelu(bn(conv2d(h0, df_dim*2, name='d_h1_conv'), 'd_bn1'))  # h1 is (64 x 64 x df_dim*2)
	h2 = lrelu(bn(conv2d(h1, df_dim*4, name='d_h2_conv'), 'd_bn2'))
	h3 = lrelu(bn(conv2d(h2, df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
	h4 = conv2d(h3, 1, s=1, name='d_h3_pred')  #shape=(?, 32, 64, 1)
	print("h0", h0)
        print("h1", h1)
	print("h2", h2)
	print("h3", h3)
	print("h4", h4)
        return h4

def refiner_cyc(img, scope, gf_dim=64, reuse=False, train=True):
    print ("refiner input",img)

    def residule_block(x, dim, name='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
	y = bn(conv2d(y, dim, 3, 1, padding='VALID', name=name+'_c1'), name+'_bn1')
	y = tf.pad(tf.nn.relu(y), [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
	y = bn(conv2d(y, dim, 3, 1, padding='VALID', name=name+'_c2'), name+'_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(bn(conv2d(c0, gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = relu(bn(conv2d(c1, gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = relu(bn(conv2d(c2, gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))

        # define G network with 9 resnet blocks
        r1 = residule_block(c3, gf_dim*4, name='g_r1')
        r2 = residule_block(r1, gf_dim*4, name='g_r2')
        r3 = residule_block(r2, gf_dim*4, name='g_r3')
        r4 = residule_block(r3, gf_dim*4, name='g_r4')
        r5 = residule_block(r4, gf_dim*4, name='g_r5')
        r6 = residule_block(r5, gf_dim*4, name='g_r6')
        r7 = residule_block(r6, gf_dim*4, name='g_r7')
        #r8 = residule_block(r7, gf_dim*4, name='g_r8')
        #r9 = residule_block(r8, gf_dim*4, name='g_r9')

        d1 = deconv2d(r7, gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = relu(bn(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc')
        d2 = relu(bn(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c'))

        print ("refiner output",pred)
        return pred

def discriminator_global(img, scope, df_dim=64, reuse=False, train=True):

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        h0 = lrelu(conv2d(img, df_dim, name='d_h0_conv')) # (?, 128, 256, 64)
        h1 = lrelu(bn(conv2d(h0, df_dim*2, name='d_h1_conv'), 'd_bn1'))  # (?, 64, 128, 128)
        h2 = lrelu(bn(conv2d(h1, df_dim*4, name='d_h2_conv'), 'd_bn2')) # (?, 32, 64, 256)
        h3 = lrelu(bn(conv2d(h2, df_dim*8, s=1, name='d_h3_conv'), 'd_bn3')) #(?, 32, 64, 512)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')  #shape=(?, 32, 64, 1)

        # flatten
        h_local = slim.flatten(h4)
	h_local = slim.fully_connected(h_local, 512,
                                              activation_fn=None,
                                              scope='h_local')
        print("h1", h1)
        print("h2", h2)
        print("h3", h3)
        print("h4", h4)
        print("h_local", h_local)
	
	h5 = slim.dropout(h3, 0.4, scope='dropout3')
        h5 = lrelu(bn(conv2d(h5, df_dim*16, name='d_h5_conv'), 'd_bn5')) #(?, 16, 32, 1024)
	h5 = slim.dropout(h5, 0.4, scope='dropout5')
	h6 = lrelu(bn(conv2d(h5, df_dim*32, name='d_h6_conv'), 'd_bn6')) #(?, 8, 16, 2048)
	h6 = slim.dropout(h6, 0.4, scope='dropout6')
	h7 = lrelu(bn(conv2d(h6, df_dim*64, name='d_h7_conv'), 'd_bn7'))
	h7 = slim.dropout(h7, 0.4, scope='dropout7')
        h_global = slim.flatten(h7)
        h_global = slim.fully_connected(h_global, 512,
                                              activation_fn=None,
                                              scope='h_global')
	h_global = slim.dropout(h_global, 0.3, scope='dropout1')
        h_global = slim.fully_connected(h_global, 16,
                                              activation_fn=None,
                                              scope='h_global2')
	#h_global = tf.nn.softmax(h_global, name='Predictions')
        h_concat = tf.concat([h_local, h_global], axis=1)

        print("h5", h5)
        print("h6", h6)
        print("h_global", h_global)
        print("h_concat", h_concat)
        return h_concat


