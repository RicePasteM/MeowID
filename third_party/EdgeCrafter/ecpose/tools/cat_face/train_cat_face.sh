#!/usr/bin/env bash
# 猫脸关键点 ECPose-X 微调训练脚本

set -euo pipefail

# 设置环境
source /home/lihb/Software/miniconda3/etc/profile.d/conda.sh
conda activate rio_aitodr

cd /home/lihb/Github/hzc_cat/EdgeCrafter/ecpose

GPU_IDS=${GPU_IDS:-0}
GPU_NUM=${GPU_NUM:-$(awk -F, '{print NF}' <<< "${GPU_IDS}")}
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# 配置参数
CONFIG=configs/ecpose/ecpose_x_cat_face.yml
PRETRAINED=${PRETRAINED:-outputs/ecpose_x_cat_face/checkpoint.pth}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/ecpose_x_cat_face_mixed}
# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 单卡训练
if [ ${GPU_NUM} -eq 1 ]; then
    python train.py \
        -c ${CONFIG} \
        -t ${PRETRAINED} \
        --output-dir ${OUTPUT_DIR} \
        --use-amp \
        2>&1 | tee ${OUTPUT_DIR}/train.log
else
    # 多卡训练
    torchrun \
        --nproc_per_node=${GPU_NUM} \
        train.py \
        -c ${CONFIG} \
        -t ${PRETRAINED} \
        --output-dir ${OUTPUT_DIR} \
        --use-amp \
        2>&1 | tee ${OUTPUT_DIR}/train.log
fi
