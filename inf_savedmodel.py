import os,time,cv2, sys, math
import tensorflow as tf
from tensorflow.python.framework import graph_util
import tfcoreml as tf_converter
import argparse
import numpy as np
import glob

from utils import utils, helpers
from builders import model_builder


def listup_files(path):
    for p in glob.glob(path + "/*"):
        yield p

# ラベルの読み込み
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

# メインの実行
if __name__ == '__main__':
    # 引数のパース
    export_dir = "growing_1_30_aug_dm025_img256_checkpoints/latest_model_MobileUNetSmall-Skip_annotated_growing_1-30-skin-eye-lips/SavedModel"
    img_height = 256
    img_width = 256

    class_names_list, label_values = helpers.get_label_info(os.path.join("annotated_growing_1-30-skin-eye-lips/", "class_dict.csv"))

    num_classes = len(label_values)

    with tf.Session(graph=tf.Graph()) as sess:
        for image in listup_files("user_test"):
            if ".csv" in image:
                continue
            print("Testing image " + image)
            loaded_image = utils.load_test_image(image)
            input_image = np.expand_dims(np.float32(cv2.resize(loaded_image, (img_height, img_width))),axis=0)/255.0

            # saved_model load
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

            net_input = sess.graph.get_tensor_by_name('input:0')
            logits = sess.graph.get_tensor_by_name('logits/BiasAdd:0')

            output_image = sess.run(logits, feed_dict={net_input: input_image})

            output_image = np.array(output_image[0,:,:,:])
            
            output_image = helpers.reverse_one_hot(output_image)

            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            file_name = utils.filepath_to_name(image)
            save_original_path = os.path.join("user_test", "%s.jpg"%(file_name))
            print("Wrote image " + "%s"%(save_original_path))
            # cv2.imwrite(save_original_path, loaded_image)
            save_predict_path = os.path.join("tf2_savedmodel_test", "%s_pred.png"%(file_name))
            print("Wrote image " + "%s"%(save_predict_path))
            cv2.imwrite(save_predict_path, np.uint8(out_vis_image))
