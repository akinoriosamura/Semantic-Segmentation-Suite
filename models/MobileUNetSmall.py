import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv2d(net, stride, channel, kernel, depth=1, scope=None):
    """
    Builds the conv block for MobileNets
    Apply successivly a 2D convolution, BatchNormalization relu
    """
    num_channel = depth(channel)
    net = slim.conv2d(net, num_channel, [kernel, kernel], stride=stride, scope=scope)
    print(net.name, net.get_shape())

    return net

def invertedresidual(net, stride, up_sample, channel, depth=1, scope=None):
    prev_output = net
    net = slim.conv2d(
        net,
        up_sample * net.get_shape().as_list()[-1],
        [1, 1],
        scope=scope + '/conv2d_1'
        )
    print(net.name, net.get_shape())
    net = slim.separable_conv2d(net, None, [3, 3],
                                depth_multiplier=1,
                                stride=stride,
                                scope=scope + '/separable2d')
    print(net.name, net.get_shape())
    num_channel = depth(channel)
    net = slim.conv2d(
        net,
        num_channel,
        [1, 1],
        activation_fn=None,
        scope=scope + '/conv2d_2'
        )
    print(net.name, net.get_shape())

    if stride == 1:
        if prev_output.get_shape().as_list(
        )[-1] != net.get_shape().as_list()[-1]:
            # Assumption based on previous ResNet papers: If the number of filters doesn't match,
            # there should be a conv 1x1 operation.
            # reference(pytorch) :
            # https://github.com/MG2033/MobileNet-V2/blob/master/layers.py#L29
            prev_output = slim.conv2d(
                prev_output,
                num_channel,
                [1, 1],
                activation_fn=None,
                biases_initializer=None,
                scope=scope + '/conv2d_3'
                )
            print(net.name, net.get_shape())

        # as described in Figure 4.
        net = tf.add(prev_output, net, name=scope + '/add')
        print(net.name, net.get_shape())
    return net


def conv_transpose_block(net, channel, kernel, stride, depth, scope=None):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    num_channel = depth(channel)
    net = slim.conv2d_transpose(net, num_channel, kernel_size=[kernel, kernel], stride=[stride, stride], padding='SAME', activation_fn=None, scope=scope+'/conv_trans')
    print(net.name, net.get_shape())
    return net

def build_mobile_unet_small(inputs, preset_model, num_classes, is_training):

    has_skip = True

    batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None,  # tf.GraphKeys.UPDATE_OPS,
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    'is_training': is_training
    }
    weight_decay = 5e-5

    depth_multi = 0.25
    def depth(d):
        min_depth=8
        return max(int(d * depth_multi), min_depth)
    
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose],
        activation_fn=tf.nn.relu6,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        biases_initializer=tf.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        padding='SAME'
        ):
        #####################
        # Downsampling path #
        #####################
        print(inputs.name, inputs.get_shape())
        # inputs -> 16
        net = invertedresidual(inputs, stride=1, up_sample=1, channel=16, depth=depth, scope='net1/inresidual')
        skip_1 = net

        # 16 -> 24
        net = invertedresidual(net, stride=2, up_sample=6, channel=24, depth=depth, scope='net2/inresidual1')
        net = invertedresidual(net, stride=1, up_sample=6, channel=24, depth=depth, scope='net2/inresidual2')
        skip_2 = net

        # 24 -> 32
        net = invertedresidual(net, stride=2, up_sample=6, channel=32, depth=depth, scope='net3/inresidual1')
        net = invertedresidual(net, stride=1, up_sample=6, channel=32, depth=depth, scope='net3/inresidual2')
        net = invertedresidual(net, stride=1, up_sample=6, channel=32, depth=depth, scope='net3/inresidual3')
        skip_3 = net

        # 32 -> 64
        net = invertedresidual(net, stride=2, up_sample=6, channel=64, depth=depth, scope='net4/inresidual1')
        net = invertedresidual(net, stride=1, up_sample=6, channel=64, depth=depth, scope='net4/inresidual2')
        net = invertedresidual(net, stride=1, up_sample=6, channel=64, depth=depth, scope='net4/inresidual3')
        net = invertedresidual(net, stride=1, up_sample=6, channel=64, depth=depth, scope='net4/inresidual4')
        # 64 -> 96
        net = invertedresidual(net, stride=1, up_sample=6, channel=96, depth=depth, scope='net5/inresidual1')
        net = invertedresidual(net, stride=1, up_sample=6, channel=96, depth=depth, scope='net5/inresidual2')
        net = invertedresidual(net, stride=1, up_sample=6, channel=96, depth=depth, scope='net5/inresidual3')
        skip_4 = net

        # 96 -> 160
        net = invertedresidual(net, stride=2, up_sample=6, channel=160, depth=depth, scope='net6/inresidual1')
        net = invertedresidual(net, stride=1, up_sample=6, channel=160, depth=depth, scope='net6/inresidual2')
        net = invertedresidual(net, stride=1, up_sample=6, channel=160, depth=depth, scope='net6/inresidual3')
        # 160 -> 320
        net = invertedresidual(net, stride=1, up_sample=6, channel=320, depth=depth, scope='net7/inresidual1')
        # 320 -> 1280
        net = conv2d(net, stride=1, channel=1280, kernel=1, depth=depth, scope='net8/conv')
        skip_5 = net

        #####################
        # Upsampling path #
        #####################
        # 1280 -> 96
        net = conv_transpose_block(net, 96, kernel=4, stride=2, depth=depth, scope='upnet1/conv_trans1')
        if has_skip:
            print("skip 1")
            print(net.name, net.get_shape())
            print(skip_4.name, skip_4.get_shape())
            net = tf.add(net, skip_4, name='upnet1/add')
            print(net.name, net.get_shape())
        net = invertedresidual(net, stride=1, up_sample=6, channel=96, depth=depth, scope='upnet1/inresidual')

        # 96 -> 32
        net = conv_transpose_block(net, 32, kernel=4, stride=2, depth=depth, scope='upnet2/conv_trans1')
        if has_skip:
            print("skip 2")
            print(net.name, net.get_shape())
            print(skip_3.name, skip_3.get_shape())
            net = tf.add(net, skip_3, name='upnet2/add')
            print(net.name, net.get_shape())
        net = invertedresidual(net, stride=1, up_sample=6, channel=32, depth=depth, scope='upnet2/inresidual')

        # 32 -> 24
        net = conv_transpose_block(net, 24, kernel=4, stride=2, depth=depth, scope='upnet3/conv_trans1')
        if has_skip:
            print("skip 3")
            print(net.name, net.get_shape())
            print(skip_2.name, skip_2.get_shape())
            net = tf.add(net, skip_2, name='upnet3/add')
            print(net.name, net.get_shape())
        net = invertedresidual(net, stride=1, up_sample=6, channel=24, depth=depth, scope='upnet3/inresidual')

        # 32 -> 24
        net = conv_transpose_block(net, 16, kernel=4, stride=2, depth=depth, scope='upnet4/conv_trans1')
        if has_skip:
            print("skip 4")
            print(net.name, net.get_shape())
            print(skip_1.name, skip_1.get_shape())
            net = tf.add(net, skip_1, name='upnet4/add')
            print(net.name, net.get_shape())
        net = invertedresidual(net, stride=1, up_sample=6, channel=16, depth=depth, scope='upnet4/inresidual')

    #####################
    #      Softmax      #
    #####################
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    print(net.name, net.get_shape())
    print("===========finish create network-------------")

    return net

