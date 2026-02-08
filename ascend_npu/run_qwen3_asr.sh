#!/bin/bash

# Qwen3-ASR-1.7B 昇腾 NPU 快速启动脚本

set -e

# 配置变量（提高可维护性）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE_SCRIPT="${SCRIPT_DIR}/activate_vllm_npu.sh"
DEMO_DIR="${SCRIPT_DIR}"

echo "======================================================================"
echo "Qwen3-ASR-1.7B 昇腾 NPU 快速启动"
echo "======================================================================"
echo ""

# 1. 激活环境
echo "[1/3] 激活 vllm_npu 环境..."
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
    echo "错误: 激活脚本不存在: ${ACTIVATE_SCRIPT}"
    exit 1
fi
source "${ACTIVATE_SCRIPT}"

# 2. 检查 NPU
echo ""
echo "[2/3] 检查 NPU 设备..."
if ! command -v npu-smi &> /dev/null; then
    echo "警告: npu-smi 命令未找到，跳过NPU检查"
else
    npu-smi info | head -15 || echo "警告: npu-smi 命令执行失败"
fi

# 3. 运行示例
echo ""
echo "[3/3] 运行测试脚本..."
echo ""
cd "${DEMO_DIR}"

case "$1" in
    demo)
        echo "运行完整演示..."
        python demo_qwen3_asr_npu.py
        ;;
    test)
        echo "运行基础测试..."
        python test_qwen3_asr_npu.py
        ;;
    ""|*)
        echo "使用方法:"
        echo "  $(basename "$0") test    # 运行基础测试"
        echo "  $(basename "$0") demo    # 运行完整演示"
        echo ""
        echo "默认运行基础测试..."
        python test_qwen3_asr_npu.py
        ;;
esac

echo ""
echo "======================================================================"
if [ $? -eq 0 ]; then
    echo "✨ 完成！"
else
    echo "⚠️  脚本执行完成，但可能有错误"
fi
echo "======================================================================"
