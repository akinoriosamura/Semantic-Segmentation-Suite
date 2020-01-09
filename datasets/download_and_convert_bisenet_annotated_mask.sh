
# prepare
"""
HOME / datasets / download....sh
                  convert....py
     / processing dataset / img / ....jpg
                          / mask / ....png(白黒の)
"""

set -e

CURRENT_DIR=$(pwd)
WORK_DIR="../"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

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
python ./convert_bisenet_annotated_mask_skin_eye_lips.py  \
  --image_folder="${GROWING_MASK_ROOT}/img" \
  --image_label_folder="${GROWING_MASK_ROOT}/mask" \
  --mask_folder="${GROWING_MASK_CREATED}/mask" \
  --output_dir="${GROWING_MASK_CREATED}"
