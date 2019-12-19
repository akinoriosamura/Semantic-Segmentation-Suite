from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os,time,cv2, sys, math
import glob

from PIL import Image

from tensorflow.lite.python.interpreter import Interpreter

from utils import utils, helpers

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_imgs', type=str, default=None, required=True, help='The image you want to predict on. ')
    parser.add_argument('--model', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
    parser.add_argument('--model_dir', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
    parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
    args = parser.parse_args()

    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

    num_classes = len(label_values)


    # インタプリタの生成
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # 入力情報と出力情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 入力テンソル種別の取得(Floatingモデルかどうか)
    floating_model = input_details[0]['dtype'] == np.float32

    # 幅と高さの取得(NxHxWxC, H:1, W:2)
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    total_t = 0
    count = 0
    for image in listup_files(args.predict_imgs):
        if ".csv" in image:
            continue
        print("Testing image " + image)

        # 入力画像のリサイズ
        loaded_image = utils.load_test_image(image)
        input_image = np.expand_dims(np.float32(cv2.resize(loaded_image, (width, height))),axis=0)/255.0

        # 入力をインタプリタに指定
        interpreter.set_tensor(input_details[0]['index'], input_image)

        # 推論の実行
        interpreter.invoke()

        # 出力の取得
        import time
        st = time.time()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        el = time.time() - st
        count += 1
        print("el: ", el)
        total_t += el
        results = np.squeeze(output_data)

        output_image = helpers.reverse_one_hot(results)

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        file_name = utils.filepath_to_name(image)
        save_original_path = os.path.join(args.model_dir, "%s.jpg"%(file_name))
        # print("Wrote image " + "%s"%(save_original_path))
        # cv2.imwrite(save_original_path, img)
        save_predict_path = os.path.join("./tflite_test", "%s_pred.png"%(file_name))
        print("Wrote image " + "%s"%(save_predict_path))
        cv2.imwrite(save_predict_path, np.uint8(out_vis_image))

print("ave time: ", total_t / count)