# Script to download and preprocess the CelebAMask-HQ dataset.
#
# Usage:
#   bash ./download_and_convert_CelebAMask-HQ.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_CelebAMask-HQ_data.py
#     - download_and_convert_CelebAMask-HQ.sh
#     + CelebAMask-HQ
#       + tfrecord
#       + ADEChallengeData2016
#         + annotations
#           + training
#           + validation
#         + images
#           + training
#           + validation

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack CelebAMask-HQ dataset.
download_and_uncompress() {
  local DATASET_URL=${1}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${DATASET_URL}"
  fi
  echo "Uncompressing ${FILENAME}"
  unzip "${FILENAME}"
}

# Download the images.
DATASET_URL="https://drive.google.com/uc?export=download&confirm=oS6Z&id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv"

# download_and_uncompress "${DATASET_URL}"

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
echo "${WORK_DIR}/CelebAMask-HQ"
# Root path for CelebAMask-HQ dataset.
CELEBAMASK_HQ_ROOT="${WORK_DIR}/CelebAMask-HQ"
CELEBAMASK_HQ_CREATED="${WORK_DIR}/CelebAMask-HQ-skin-eye-lips"
mkdir -p "${CELEBAMASK_HQ_CREATED}"

# create trainig labels
# split train and val
# skin, neck, hair labels dataset
python ./convert_celebamask-hq_skin_eye_lips.py  \
  --image_folder="${CELEBAMASK_HQ_ROOT}/CelebA-HQ-img" \
  --image_label_folder="${CELEBAMASK_HQ_ROOT}/CelebAMask-HQ-mask-anno" \
  --mask_folder="${CELEBAMASK_HQ_CREATED}/mask" \
  --output_dir="${CELEBAMASK_HQ_CREATED}"
