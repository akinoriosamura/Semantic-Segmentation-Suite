import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], scope=None):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size=[1, 1], activation_fn=None, scope=scope+'/conv2d')
	net = slim.batch_norm(net, fused=True, scope=scope+'/bn')
	net = tf.nn.relu(net, name=scope+'/relu')
	return net

def DepthwiseSeparableConvBlock(inputs, n_filters, kernel_size=[3, 3], scope=None):
	"""
	Builds the Depthwise Separable conv block for MobileNets
	Apply successivly a 2D separable convolution, BatchNormalization relu, conv, BatchNormalization, relu
	"""
	# Skip pointwise by setting num_outputs=None
	net = slim.separable_convolution2d(inputs, num_outputs=None, depth_multiplier=1, kernel_size=[3, 3], activation_fn=None, scope=scope+'/sepa_conv2d')

	net = slim.batch_norm(net, fused=True, scope=scope+'/bn1')
	net = tf.nn.relu(net, name=scope+'/relu1')
	net = slim.conv2d(net, n_filters, kernel_size=[1, 1], activation_fn=None, scope=scope+'/conv2d')
	net = slim.batch_norm(net, fused=True, scope=scope+'/bn2')
	net = tf.nn.relu(net, name=scope+'/relu2')
	return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], scope=None):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None, scope=scope+'/conv_trans')
	net = tf.nn.relu(slim.batch_norm(net, scope=scope+'/bn'), name=scope+'/relu')
	return net

def build_mobile_unet(inputs, preset_model, num_classes, is_training):

	has_skip = False
	if preset_model == "MobileUNet":
		has_skip = False
	elif preset_model == "MobileUNet-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))

    #####################
	# Downsampling path #
	#####################
	net = ConvBlock(inputs, 64, scope='net1/conv1')
	net = DepthwiseSeparableConvBlock(net, 64, scope='net1/depthconv1')
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='net1/maxpool1')
	skip_1 = net

	net = DepthwiseSeparableConvBlock(net, 128, scope='net2/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 128, scope='net2/depthconv2')
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='net2/maxpool1')
	skip_2 = net

	net = DepthwiseSeparableConvBlock(net, 256, scope='net3/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 256, scope='net3/depthconv2')
	net = DepthwiseSeparableConvBlock(net, 256, scope='net3/depthconv3')
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='net3/maxpool1')
	skip_3 = net

	net = DepthwiseSeparableConvBlock(net, 512, scope='net4/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 512, scope='net4/depthconv2')
	net = DepthwiseSeparableConvBlock(net, 512, scope='net4/depthconv3')
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='net4/maxpool1')
	skip_4 = net

	net = DepthwiseSeparableConvBlock(net, 512, scope='net5/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 512, scope='net5/depthconv2')
	net = DepthwiseSeparableConvBlock(net, 512, scope='net5/depthconv3')
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX', scope='net5/maxpool1')


	#####################
	# Upsampling path #
	#####################
	net = conv_transpose_block(net, 512, scope='upnet1/conv_trans1')
	net = DepthwiseSeparableConvBlock(net, 512, scope='upnet1/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 512, scope='upnet1/depthconv2')
	net = DepthwiseSeparableConvBlock(net, 512, scope='upnet1/depthconv3')
	if has_skip:
		net = tf.add(net, skip_4, name='upnet1/add')

	net = conv_transpose_block(net, 512, scope='upnet2/conv_trans1')
	net = DepthwiseSeparableConvBlock(net, 512, scope='upnet2/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 512, scope='upnet2/depthconv2')
	net = DepthwiseSeparableConvBlock(net, 256, scope='upnet2/depthconv3')
	if has_skip:
		net = tf.add(net, skip_3, name='upnet2/add')

	net = conv_transpose_block(net, 256, scope='upnet3/conv_trans1')
	net = DepthwiseSeparableConvBlock(net, 256, scope='upnet3/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 256, scope='upnet3/depthconv2')
	net = DepthwiseSeparableConvBlock(net, 128, scope='upnet3/depthconv3')
	if has_skip:
		net = tf.add(net, skip_2, name='upnet3/add')

	net = conv_transpose_block(net, 128, scope='upnet4/conv_trans1')
	net = DepthwiseSeparableConvBlock(net, 128, scope='upnet4/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 64, scope='upnet4/depthconv2')
	if has_skip:
		net = tf.add(net, skip_1, name='upnet4/add')

	net = conv_transpose_block(net, 64, scope='upnet5/conv_trans1')
	net = DepthwiseSeparableConvBlock(net, 64, scope='upnet5/depthconv1')
	net = DepthwiseSeparableConvBlock(net, 64, scope='upnet5/depthconv2')

	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	print(net.name, net.get_shape())
	print("===========finish create network")

	return net

