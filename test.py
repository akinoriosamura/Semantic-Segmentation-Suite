import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import cv2

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_or_resize', type=str, default="resize", help='crop or resize of input')
parser.add_argument('--img_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--img_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
with tf.Graph().as_default() as graph:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(
        graph=graph,
        config=config
        )

    with sess.as_default():
        net_input = tf.placeholder(tf.float32,shape=[None,args.img_height,args.img_width,3], name='input')

        network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, img_width=args.img_width, img_height=args.img_height, is_training=False)

        sess.run(tf.global_variables_initializer())

        print('Loading model checkpoint weights ...')
        saver=tf.train.Saver(max_to_keep=1000)
        saver.restore(sess, args.checkpoint_path)

        # Load the data
        print("Loading the data ...")
        train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

        # Create directories if needed
        if not os.path.isdir("%s"%("Test")):
                os.makedirs("%s"%("Test"))

        target=open("%s/test_scores.csv"%("Test"),'w')
        target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        run_times_list = []

        test_input_names = test_input_names[:100]

        # Run testing on ALL test images
        for ind in range(len(test_input_names)):
            sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
            sys.stdout.flush()

            input_image = utils.load_image(test_input_names[ind])
            gt = utils.load_image(test_output_names[ind])
            if args.crop_or_resize == "crop":
                input_image = np.expand_dims(np.float32(input_image[:args.img_height, :args.img_width]),axis=0)/255.0
                gt = gt[:args.img_height, :args.img_width]
            else:
                input_image = np.expand_dims(np.float32(cv2.resize(input_image, (args.img_height, args.img_width))),axis=0)/255.0
                gt = cv2.resize(gt, (args.img_height, args.img_width))

            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            st = time.time()
            output_image = sess.run(network,feed_dict={net_input:input_image})

            run_times_list.append(time.time()-st)

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

            file_name = utils.filepath_to_name(test_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            
            gt = helpers.colour_code_segmentation(gt, label_values)

            cv2.imwrite("%s/%s_pred.png"%("Test", file_name),np.uint8(out_vis_image))
            cv2.imwrite("%s/%s_gt.png"%("Test", file_name),np.uint8(gt))


        target.close()

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)
print("Average test accuracy = ", avg_score)
print("Average per class test accuracies = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("Average precision = ", avg_precision)
print("Average recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average mean IoU score = ", avg_iou)
print("Average run time = ", avg_time)
