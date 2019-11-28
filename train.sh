# python train.py --dataset=CamVid/ --batch_size=1 --crop_or_resize=crop
python train.py --dataset=CelebAMask-HQ-skin-eye-lips/ \
	--batch_size=1 \
	--crop_or_resize=resize \
	--model=MobileUNet-Skip \
	--frontend=MobileNetV2
