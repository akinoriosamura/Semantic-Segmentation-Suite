#!/bin/bash
phase=$1
depth_multi=0.25 # default = 1, like model complicated
num_quant=64
dataset=annotated_growing_1-30-skin-eye-lips/
batch_size=1
crop_or_resize=resize
img_size=256
model=MobileUNetSmall-Skip
frontend=MobileNetV2
# data augment
h_flip=True
v_flip=True
brightness=0.5  # Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).
rotation=90  # Specifies the max rotation angle in degrees.

# ModelDir="aug_dm025_img256_checkpoints/latest_model_MobileUNetSmall-Skip_CelebAMask-HQ-skin-eye-lips/"
ModelDir="growing_1_30_aug_dm025_img256_checkpoints/latest_model_MobileUNetSmall-Skip_annotated_growing_1-30-skin-eye-lips/"
CheckPoint="${ModelDir}ckpt"
tflite_model="${ModelDir}tf2_mobile_unet.tflite"

predict_dir="user_test"

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_hlo_profile

if [ ${phase} = "train" ]; then
    echo "run train"
    python train.py --dataset=${dataset} \
                    --batch_size=${batch_size} \
                    --img_height=${img_size} \
                    --img_width=${img_size} \
                    --crop_or_resize=${crop_or_resize} \
                    --model=${model} \
                    --frontend=${frontend} \
                    --h_flip=${h_flip} \
                    --v_flip=${v_flip} \
                    --brightness=${brightness} \
                    --rotation=${rotation}

elif [ ${phase} = "pretrain" ]; then
    echo "run pretrain"
    python train.py --dataset=${dataset} \
                    --batch_size=${batch_size} \
                    --img_height=${img_size} \
                    --img_width=${img_size} \
                    --crop_or_resize=${crop_or_resize} \
                    --model_checkpoint_path=${CheckPoint} \
                    --model=${model} \
                    --frontend=${frontend} \
                    --h_flip=${h_flip} \
                    --v_flip=${v_flip} \
                    --brightness=${brightness} \
                    --rotation=${rotation}

elif [ ${phase} = "save" ]; then
    echo "run save"
    python predict_save.py --dataset=${dataset} \
                            --crop_or_resize=${crop_or_resize} \
                            --img_height=${img_size} \
                            --img_width=${img_size} \
                            --model=${model} \
                            --model_dir=${ModelDir} \
                            --checkpoint_path=${CheckPoint} \
                            --predict_imgs=${predict_dir}

elif [ "${phase}" = 'test' ]; then
    echo "run test"
    python test.py --dataset=${dataset} \
                    --crop_or_resize=${crop_or_resize} \
                    --img_height=${img_size} \
                    --img_width=${img_size} \
                    --model=${model} \
                    --checkpoint_path=${CheckPoint}

elif [ "${phase}" = 'tflite' ]; then
    echo "run tflite"
    python inf_tflite.py --dataset=${dataset} \
                    --model_dir=${ModelDir} \
                    --model=${tflite_model} \
                    --predict_imgs=${predict_dir}
else
    echo "no running"
fi
