ModelDir="checkpoints/latest_model_MobileUNetSmall-Skip_CelebAMask-HQ-skin-eye-lips/"
CheckPoint="${ModelDir}ckpt"

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_hlo_profile
python predict_save.py --dataset=CelebAMask-HQ-skin-eye-lips/ \
	--crop_or_resize=resize \
	--model=MobileUNetSmall-Skip \
    --model_dir="${ModelDir}" \
	--checkpoint_path="${CheckPoint}" \
	--image=CelebAMask-HQ-skin-eye-lips/test/10000.png
