set -e

CURRENT_DIR=$(pwd)
WORK_DIR="../"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# =================== full labels ===================
# cd "${CURRENT_DIR}"
# echo "${CURRENT_DIR}"
# echo "${WORK_DIR}/CelebAMask-HQ"
# # Root path for CelebAMask-HQ dataset.
# CELEBAMASK_HQ_ROOT="${WORK_DIR}/CelebAMask-HQ"
# 
# # create trainig labels
# # split train and val
# # full labels dataset
# python ./convert_celebamask-hq.py  \
#   --image_folder="${CELEBAMASK_HQ_ROOT}/CelebA-HQ-img" \
#   --image_label_folder="${CELEBAMASK_HQ_ROOT}/CelebAMask-HQ-mask-anno" \
#   --mask_folder="${CELEBAMASK_HQ_ROOT}/mask" \
#   --output_dir="${CELEBAMASK_HQ_ROOT}"
# 
# # Build TFRecords of the dataset.
# # First, create output directory for storing TFRecords.
# OUTPUT_DIR="${WORK_DIR}/tfrecord"
# mkdir -p "${OUTPUT_DIR}"
# 
# echo "Converting CelebAMask-HQ dataset..."
# python ./build_CelebAMask-HQ_data.py  \
#   --train_image_folder="${CELEBAMASK_HQ_ROOT}/images/train/" \
#   --train_image_label_folder="${CELEBAMASK_HQ_ROOT}/annotations/train/" \
#   --val_image_folder="${CELEBAMASK_HQ_ROOT}/images/val/" \
#   --val_image_label_folder="${CELEBAMASK_HQ_ROOT}/annotations/val/" \
#   --output_dir="${WORK_DIR}/tfrecord" \

# =================== small labels ===================
# skin, neck, hair labels dataset
cd "${CURRENT_DIR}"
echo "${CURRENT_DIR}"
echo "${WORK_DIR}/annotated_growing_1-30"
# Root path for annotated_growing_1-30 dataset.
GROWING_MASK_ROOT="${WORK_DIR}/annotated_growing_1-30"
GROWING_MASK_CREATED="${WORK_DIR}/annotated_growing_1-30-skin-eye-lips"
mkdir -p "${GROWING_MASK_CREATED}"

# create trainig labels
# split train and val
# skin, neck, hair labels dataset
python ./convert_growing_mask_skin_eye_lips.py  \
  --image_folder="${GROWING_MASK_ROOT}/img" \
  --image_label_folder="${GROWING_MASK_ROOT}/mask" \
  --mask_folder="${GROWING_MASK_CREATED}/mask" \
  --output_dir="${GROWING_MASK_CREATED}"
