#!/bin/bash
# 一键运行脚本：数据预处理 -> 训练 -> 测试

set -e  # 遇到错误立即退出

echo "======================================"
echo "WebQ检索系统 - 完整流程"
echo "======================================"

# 设置路径
BASE_DIR="/home/tanyuqiao/IR_Project-main/data/IR_2025_Project"
cd $BASE_DIR

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: python not found"
    exit 1
fi

# 安装依赖
echo ""
echo "Step 0: 检查并安装依赖..."
pip install transformers torch tqdm -q 2>/dev/null || echo "Dependencies already installed"

# Step 1: 数据预处理
echo ""
echo "======================================"
echo "Step 1: 数据预处理"
echo "======================================"
python preprocess_data.py

# 检查数据预处理是否成功
if [ ! -f "$BASE_DIR/processed/passages.pkl" ]; then
    echo "Error: 数据预处理失败，找不到passages.pkl"
    exit 1
fi

echo "✓ 数据预处理完成！"

# Step 2: 训练模型
echo ""
echo "======================================"
echo "Step 2: 训练检索模型"
echo "======================================"
python train_simple.py

# 检查训练是否成功
if [ ! -f "$BASE_DIR/output/retriever/best_model.pt" ]; then
    echo "Error: 训练失败，找不到best_model.pt"
    exit 1
fi

echo "✓ 训练完成！"

# Step 3: 测试模型
echo ""
echo "======================================"
echo "Step 3: 在测试集上评估"
echo "======================================"
python test_simple.py

echo ""
echo "======================================"
echo "✓ 全部完成！"
echo "======================================"
echo ""
echo "结果保存在: ./output/test_results/"
echo ""
echo "查看结果:"
echo "  - cat ./output/test_results/metrics.json"
echo "  - cat ./output/test_results/hit_at_k.txt"
echo ""
echo "提交材料:"
echo "  1. 源代码: *.py"
echo "  2. 结果文件: ./output/test_results/hit_at_k.txt"
echo "  3. 实验报告: 需要单独撰写"
echo ""
