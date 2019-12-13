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

# by face parsing
# BGR設定
BASE_PART_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], #3
            [255, 0, 85], [255, 0, 170],[0, 255, 0], #6
            [85, 255, 0], [170, 255, 0], [0, 255, 85], #9
            [0, 255, 170], [0, 0, 255], [85, 0, 255], #12
            [170, 0, 255], [0, 85, 255], [0, 170, 255], #15
            [255, 255, 0], [255, 255, 85], [255, 255, 170], #18
            [255, 0, 255], [255, 85, 255], [255, 170, 255], #21
            [0, 255, 255], [85, 255, 255], [170, 255, 255]] #24


def create_base(face_data, face_sep_mask):
    p_images = pathlib.Path(face_data)
    images = list(p_images.glob('**/*.jpg'))
    base_images = [os.path.basename(str(p_img))[:-4] for p_img in images]
    p_annos = pathlib.Path(face_sep_mask)
    annotations = list(p_annos.glob('**/*.png'))
    base_annotations = [os.path.basename(str(p_anno))[:-4] for p_anno in annotations]
    base_paths = list(set(base_images) & set(base_annotations))

    return base_paths


def create_images_dataset(imgs_p, saved_dir):
    print("processing : ", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    for im_p in imgs_p:
        im_p = str(im_p)
        img = Image.open(str(im_p))
        # img = np.array(img.resize((512, 512), Image.BILINEAR))
        im_p = str(im_p)[:-4] + ".png"
        im_base = os.path.basename(im_p)
        img.save('{}/{}'.format(saved_dir, im_base))


def create_annotations_dataset(annos_p, saved_dir):
    print("processing : ", saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    for anno_p in annos_p:
        anno = Image.open(str(anno_p))
        # print(np.array(anno).shape)
        anno_base = os.path.basename(anno_p)
        anno.save('{}/{}'.format(saved_dir, anno_base))


def convert_annotations(base_paths, face_data, face_sep_mask, mask_path):
    # convet color map to pallete mask
    counter = 0
    total = 0
    for id, base_p in enumerate(base_paths):
        if id % 100 == 0:
            print("process img :", id)

        original_atts = {
            'skin':[255, 85, 0],
            'nose': [0, 0, 255],
            'r_eyebrow': [255, 0, 85],
            'l_eyebrow': [255, 170, 0],
            'l_eye': [0, 255, 0],
            'r_eye': [255, 0, 170],
            'u_lip': [170, 0, 255],
            'l_lip': [0, 85, 255],
            'mouth': [85, 0, 255]
        }

        atts = {
            'skin':[255, 85, 0],
            'nose': [255, 85, 0],
            'r_eyebrow': [255, 85, 0],
            'l_eyebrow': [255, 85, 0],
            'l_eye': [0, 255, 0],
            'r_eye': [0, 255, 0],
            'u_lip': [0, 85, 255],
            'l_lip': [0, 85, 255],
            'mouth': [0, 0, 255]
            }

        target_img = cv2.imread(os.path.join(face_data, base_p + ".jpg"))
        anno_mask = np.array(cv2.imread(osp.join(face_sep_mask, base_p + ".png")))
        anno_mask = cv2.resize(anno_mask, (target_img.shape[1], target_img.shape[0]))
        anno = np.zeros((anno_mask.shape[0], anno_mask.shape[1], 3)) + 255
        num_of_class = np.max(anno_mask)
        # import pdb;pdb.set_trace()
        for pi in range(1, num_of_class + 1):
            # index = np.where((anno_mask == pi))
            # anno[index[0], index[1], :] = BASE_PART_COLORS[pi]
            anno[np.where((anno_mask == [pi, pi, pi]).all(axis=2))] = BASE_PART_COLORS[pi]
        # cv2.imwrite('{}/{}.png'.format(mask_path, base_p), anno)
        anno = anno.astype(np.uint8)

        sep_mask = np.zeros(target_img.shape)

        # add color to use attr
        for ori_att, att in original_atts.items():
            sep_mask[np.where((anno == att).all(axis=2))] = atts[ori_att]

        # opening and closing anno img
        kernel = np.ones((2,2),np.uint8)
        sep_mask = cv2.morphologyEx(sep_mask, cv2.MORPH_OPEN, kernel)
        sep_mask = cv2.morphologyEx(sep_mask, cv2.MORPH_CLOSE, kernel)

        # save mask by same raw image name but png
        cv2.imwrite('{}/{}.png'.format(mask_path, base_p), sep_mask)


def _convert_and_save_dataset(face_data, face_sep_mask, mask_path, output_dir):
    # get base names path
    base_paths = create_base(face_data, face_sep_mask)

    # save mask from anno
    convert_annotations(base_paths, face_data, face_sep_mask, mask_path)

    # get images and masks and split and save
    images = [os.path.join(face_data, base_p + ".jpg") for base_p in base_paths]
    p_masks = pathlib.Path(mask_path)
    annotations = list(p_masks.glob('**/*.png'))
    annotations = [str(anno) for anno in annotations]
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
