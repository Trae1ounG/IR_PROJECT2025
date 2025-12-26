#!/bin/bash
# 环境设置脚本
#
# 使用方法：
# source setup_env.sh

echo "激活conda环境 ir2025..."

# 激活conda环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ir2025

echo "✓ 环境已激活"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "可以使用以下GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl
