#!/bin/bash
# 数据准备脚本 - 将 CSV 转换为 COCO JSON 格式

source /home/lihb/Software/miniconda3/etc/profile.d/conda.sh
conda activate ec

cd /home/lihb/object-detection/catrec/EdgeCrafter/ecpose/tools/cat_face

DATA_DIR="/data/lihb/Datasets/catrec/aistarted_cat_face"
OUTPUT_DIR="${DATA_DIR}/annotations"

echo "=========================================="
echo "猫脸关键点数据集准备"
echo "=========================================="
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 转换数据格式
python csv_to_coco.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "数据准备完成!"
echo "=========================================="
echo ""
echo "生成的文件:"
ls -lh ${OUTPUT_DIR}/
echo ""
echo "下一步: 下载 COCO 预训练权重并开始训练"
echo "  bash train_cat_face.sh"
