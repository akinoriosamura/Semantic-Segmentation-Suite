#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""Converts CelebAMask-HQ data to TFRecord file format with Example protos."""

import os.path as osp
import os
import cv2
from PIL import Image
import pathlib
import glob
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'image_folder',
    './CelebA-HQ-img',
    'Folder containing images')
tf.app.flags.DEFINE_string(
    'image_label_folder',
    './CelebAMask-HQ-mask-anno',
    'Folder containing annotations for images')
tf.app.flags.DEFINE_string(
    'mask_folder',
    './mask',
    'Folder saving converting images')
tf.app.flags.DEFINE_string(
    'output_dir',
    './CelebAMask-HQ',
    'save images and annotations in this Folder')


def create_images_dataset(imgs_p, saved_dir):
    print("processing : ", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    for im_p in imgs_p:
        img = cv2.imread(str(im_p))
        # img = np.array(img.resize((512, 512), Image.BILINEAR))
        im_base = os.path.basename(im_p)
        cv2.imwrite('{}/{}'.format(saved_dir, im_base), img)


def create_annotations_dataset(annos_p, saved_dir):
    print("processing : ", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    for anno_p in annos_p:
        anno = np.array(cv2.imread(str(anno_p)))
        anno_base = os.path.basename(anno_p)
        cv2.imwrite('{}/{}'.format(saved_dir, anno_base), anno)


def _convert_and_save_dataset(face_data, face_sep_mask, mask_path, output_dir):
    # convet color map to pallete mask
    counter = 0
    total = 0
    for i in range(15):
        print("process dataset : ", i)

        for j in range(i * 2000, (i + 1) * 2000):
            atts = {
                'skin':[0, 0, 192],
                'l_eye': [0,195,0],
                'r_eye': [0,195,0],
                'u_lip': [128, 0, 0],
                'l_lip': [128, 0, 0]
                }

            all_atts = {
                'skin': [0, 0, 0],
                'l_brow': [0, 0, 0],
                'r_brow': [0, 0, 0],
                'l_eye': [0, 0, 0],
                'r_eye': [0, 0, 0],
                'eye_g': [0, 0, 0],
                'l_ear': [0, 0, 0],
                'r_ear': [0, 0, 0],
                'ear_r': [0, 0, 0],
                'nose': [0, 0, 0],
                'mouth': [0, 0, 0],
                'u_lip': [0, 0, 0],
                'l_lip': [0, 0, 0],
                'neck': [0, 0, 0],
                'neck_l': [0, 0, 0],
                'cloth': [0, 0, 0],
                'hair': [0, 0, 0],
                'hat': [0, 0, 0]
                }
            # mask = np.zeros((512, 512))
            target_img = cv2.imread(os.path.join(face_data, str(j)+".jpg"))
            sep_mask = np.zeros(target_img.shape)
            white = [255, 255, 255]

            # add color to use attr
            for att, att_rgb in atts.items():
                del all_atts[att]

                mask_file_name = ''.join(
                    [str(j).rjust(5, '0'), '_', att, '.png'])
                # print(mask_file_name)
                path = osp.join(face_sep_mask, str(i), mask_file_name)

                if os.path.exists(path):
                    counter += 1
                    anno = np.array(cv2.imread(path))
                    sep_mask[np.where((anno == white).all(axis=2))] = att_rgb
                    # print(np.unique(sep_mask))

            # remove unuse att from sep mask
            for unuse_att, block in all_atts.items():
                mask_file_name = ''.join(
                    [str(j).rjust(5, '0'), '_', unuse_att, '.png'])
                # print(mask_file_name)
                path = osp.join(face_sep_mask, str(i), mask_file_name)

                if os.path.exists(path):
                    counter += 1
                    anno = np.array(cv2.imread(path))
                    sep_mask[np.where((anno == white).all(axis=2))] = block

            # save mask by same raw image name but png
            cv2.imwrite('{}/{}.png'.format(mask_path, j), sep_mask)

    # get images and masks and split and save
    p_images = pathlib.Path(face_data)
    images = list(p_images.glob('**/*.jpg'))
    p_masks = pathlib.Path(mask_path)
    annotations = list(p_masks.glob('**/*.png'))
    # remove no images data
    image_bases = [os.path.basename(img)[:-4] for img in images]
    annotations = [anno for anno in annotations if os.path.basename(anno)[:-4] in image_bases]
    assert len(images) == len(annotations), "dont match length images and annotation"
    images.sort()
    annotations.sort()

    (images_train,
     images_val_test,
     annotations_train,
     annotations_val_test) = train_test_split(images,
                                         annotations,
                                         test_size=0.2,
                                         )
    (images_val,
     images_test,
     annotations_val,
     annotations_test) = train_test_split(images_val_test,
                                         annotations_val_test,
                                         test_size=0.5,
                                         )

    print("image train num: ", len(images_train))
    print("image val num: ", len(images_val))
    print("image test num: ", len(images_test))
    print("anno train num: ", len(annotations_train))
    print("anno val num: ", len(annotations_val))
    print("anno test num: ", len(annotations_test))

    images_train_dir = os.path.join(output_dir, "train")
    images_val_dir = os.path.join(output_dir, "val")
    images_test_dir = os.path.join(output_dir, "test")
    annotations_train_dir = os.path.join(output_dir, "train_labels")
    annotations_val_dir = os.path.join(output_dir, "val_labels")
    annotations_test_dir = os.path.join(output_dir, "test_labels")

    create_images_dataset(images_train, images_train_dir)
    create_images_dataset(images_val, images_val_dir)
    create_images_dataset(images_test, images_test_dir)

    create_annotations_dataset(annotations_train, annotations_train_dir)
    create_annotations_dataset(annotations_val, annotations_val_dir)
    create_annotations_dataset(annotations_test, annotations_test_dir)

    print("finish save images")


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.mask_folder)
    _convert_and_save_dataset(
        FLAGS.image_folder,
        FLAGS.image_label_folder,
        FLAGS.mask_folder,
        FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
