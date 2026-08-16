#!/bin/bash
# 下载 ECPose-X 预训练权重

cd /home/lihb/object-detection/catrec/EdgeCrafter/ecpose

# 创建权重目录
mkdir -p weights
mkdir -p ecvits

echo "=========================================="
echo "下载 ECPose-X 预训练权重"
echo "=========================================="

# 下载 ECPose-X 权重 (从 GitHub Releases)
echo "下载 ECPose-X 权重..."
wget -c https://github.com/Intellindust-AI-Lab/EdgeCrafter/releases/download/edgecrafterv1/ecpose_x.pth -O weights/ecpose_x.pth

# 下载 ViTAdapter backbone 权重 (如果需要)
echo "下载 ViTAdapter backbone 权重..."
wget -c https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_vitsplus.pth -O ecvits/ecpose_vitsplus.pth

echo ""
echo "=========================================="
echo "权重下载完成!"
echo "=========================================="
ls -lh weights/
ls -lh ecvits/
