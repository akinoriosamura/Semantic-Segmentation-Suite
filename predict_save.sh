ModelDir="checkpoints/latest_model_MobileUNet-Skip_CelebAMask-HQ-skin-eye-lips/"
CheckPoint="${ModelDir}ckpt"

python predict_save.py --dataset=CelebAMask-HQ-skin-eye-lips/ \
	--crop_or_resize=resize \
	--model=MobileUNet-Skip \
    --model_dir="${ModelDir}" \
	--checkpoint_path="${CheckPoint}" \
	--image=CelebAMask-HQ-skin-eye-lips/test/10000.png
