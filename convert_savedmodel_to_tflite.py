import os,time,cv2, sys, math
import tensorflow as tf
from tensorflow.python.framework import graph_util
import tfcoreml as tf_converter
import argparse
import numpy as np
import glob

from utils import utils, helpers
from builders import model_builder

def create_save_model(model_dir):
    # tflite変換
    save_model_dir = os.path.join(model_dir, "SavedModel")
    converter = tf.lite.TFLiteConverter.from_saved_model(
        save_model_dir
        )
    # converter.allow_custom_ops=True
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    with open(model_dir + "/custom_mobile_unet.tflite", 'wb') as f:
        f.write(tflite_model)
    print("finish save tflite_model")


if __name__ == '__main__':
    create_save_model("growing_1_30_aug_dm025_img256_checkpoints/latest_model_MobileUNetSmall-Skip_annotated_growing_1-30-skin-eye-lips/")
