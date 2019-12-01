import tensorflow as tf

def dice_loss(input, target):
    smooth = 1.

    #if scale is not None:
    #    scaled = interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
    #    iflat = scaled.view(-1)
    #else:
    iflat = tf.reshape(input, [-1])

    # tflat = target.view(-1)
    tflat = tf.cast(tf.reshape(target, [-1]), tf.float32)
    print((iflat * tflat).shape)
    intersection = tf.reduce_sum(iflat * tflat)

    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(iflat) + tf.reduce_sum(tflat) + smooth))
