import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder


def create_save_model(model_dir, graph, sess):
    # save graphdef file to pb
    print("Save frozen graph")
    graphdef_n = "original_frozen.pb"
    graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ["output"])
    tf.train.write_graph(graph_def, model_dir, graphdef_n, as_text=False)

    # save SavedModel
    print("get tensor")
    net_input = graph.get_tensor_by_name('input:0')
    net_output = graph.get_tensor_by_name('output:0')
    print("start save saved_model")
    save_model_dir = os.path.join(model_dir, "SavedModel")
    # tf.saved_model.simple_save(sess, save_model_dir, inputs={"image_batch": image_batch}, outputs={"pfld_inference/fc/BiasAdd": landmarks_pre})
    builder = tf.saved_model.builder.SavedModelBuilder(save_model_dir)
    signature = tf.saved_model.predict_signature_def(
        {"input": input}, outputs={"output": output}
    )

    # using custom tag instead of: tags=[tf.saved_model.tag_constants.SERVING]
    builder.add_meta_graph_and_variables(
        sess=sess, tags=[
            tf.saved_model.tag_constants.SERVING], signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    print("finish save saved_model")



parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--model_dir', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_or_resize', type=str, default="resize", help='crop or resize of input')
parser.add_argument('--img_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--img_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("img Height -->", args.crop_height)
print("img Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
with tf.Graph().as_default() as inf_g:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(
        graph=inf_g,
        config=config
        )

    with sess.as_default():
        net_input = tf.placeholder(tf.float32,shape=[None,None,None,3], name='input')
        net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes], name='output') 

        network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, img_width=args.img_width, img_height=args.img_height, is_training=False)

        sess.run(tf.global_variables_initializer())

        print('Loading model checkpoint weights')
        saver=tf.train.Saver(max_to_keep=1000)
        print('Model directory: {}'.format(args.model_dir))
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        print('ckpt: {}'.format(ckpt))
        model_path = ckpt.model_checkpoint_path
        assert (ckpt and model_path)
        epoch_start = int(
            model_path[model_path.find('model.ckpt-') + 11:]) + 1
        print('Checkpoint file: {}'.format(model_path))
        saver.restore(sess, model_path)

        print("Testing image " + args.image)

        loaded_image = utils.load_image(args.image)
        input_image = np.expand_dims(np.float32(cv2.resize(input_image, (args.img_height, args.img_width))),axis=0)/255.0

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_time = time.time()-st

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        file_name = utils.filepath_to_name(args.image)
        cv2.imwrite("%s_pred.png"%(file_name),np.uint8(out_vis_image))

        print("")
        print("Finished!")
        print("Wrote image " + "%s_pred.png"%(file_name))

        create_save_model(args.model_dir, inf_g, sess)
