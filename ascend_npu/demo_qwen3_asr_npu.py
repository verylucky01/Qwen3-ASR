"""
Qwen3-ASR-1.7B 昇腾 NPU 完整演示脚本

展示: 
1. 单个音频识别（自动检测语言）
2. 指定语言识别
3. NPU 资源监控
"""

import sys
import time
from contextlib import contextmanager

import torch
import torch_npu

# -- 配置常量 ----------------------------------------------------------------

QWEN_ASR_LIB_PATH = "/root/demo2/Qwen3-ASR"
MODEL_PATH = "/root/demo2/Qwen3-ASR-1.7B"
AUDIO_PATH = "/root/demo2/ascend_audio.mp3"

DEVICE = "npu:0"
DEVICE_INDEX = 0
DTYPE = torch.bfloat16
MAX_BATCH_SIZE = 32
MAX_NEW_TOKENS = 256

SEPARATOR_WIDTH = 70
BYTES_PER_GB = 1024 ** 3

# -- 延迟导入本地模块 ----------------------------------------------------------

sys.path.insert(0, QWEN_ASR_LIB_PATH)
from qwen_asr import Qwen3ASRModel  # noqa: E402


# -- 辅助工具 ----------------------------------------------------------------

@contextmanager
def timer():
    """轻量计时上下文管理器，通过 result.elapsed 获取耗时（秒）。"""
    result = type("TimerResult", (), {"elapsed": 0.0})()
    start = time.time()
    yield result
    result.elapsed = time.time() - start


def bytes_to_gb(value):
    """将字节数转换为 GB。"""
    return value / BYTES_PER_GB


def print_section_header(number, title):
    """打印统一格式的节标题。"""
    print(f"\n[{number}] {title}")
    print("-" * SEPARATOR_WIDTH)


def print_transcription_results(results):
    """打印转录结果列表。"""
    print("\n输出:")
    for i, result in enumerate(results):
        print(f"  [样本 {i}]")
        print(f"    语言: {result.language}")
        print(f"    识别文本: {result.text}")


# -- 核心步骤 ----------------------------------------------------------------

def check_npu_environment():
    """检查并打印 NPU 环境信息，返回设备属性。"""
    print_section_header(1, "检查 NPU 设备信息")

    if not torch_npu.npu.is_available():
        sys.exit("错误: 未检测到可用的 NPU 设备，请检查驱动和硬件配置。")

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"torch_npu 版本: {torch_npu.__version__}")
    print(f"NPU 设备数量: {torch_npu.npu.device_count()}")
    print(f"NPU 设备名称: {torch_npu.npu.get_device_name(DEVICE_INDEX)}")

    props = torch_npu.npu.get_device_properties(DEVICE_INDEX)
    print(f"NPU 设备属性:")
    print(f"  - 名称: {props.name}")
    print(f"  - 总内存: {bytes_to_gb(props.total_memory):.2f} GB")

    return props


def load_model():
    """加载 Qwen3-ASR 模型到 NPU，返回 (模型实例, 加载耗时)。"""
    print_section_header(2, "加载 Qwen3-ASR-1.7B 模型到 NPU")

    with timer() as t:
        model = Qwen3ASRModel.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            device_map=DEVICE,
            max_inference_batch_size=MAX_BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    print(f"模型加载成功！耗时: {t.elapsed:.2f} 秒")
    print(f"模型后端: {model.backend}")
    device_info = model.model.device if hasattr(model.model, "device") else DEVICE
    print(f"模型设备: {device_info}")

    return model, t.elapsed


def transcribe_audio(model, audio_path, language=None):
    """
    执行语音转录并打印结果。

    Args:
        model: 已加载的 ASR 模型实例。
        audio_path: 音频文件路径。
        language: 指定语言（None 表示自动检测）。

    Returns:
        (转录结果列表, 推理耗时)
    """
    lang_label = language if language else "自动检测"
    print(f"输入: {audio_path}")
    print(f"语言: {lang_label}")

    with timer() as t:
        results = model.transcribe(
            audio=audio_path,
            language=language,
            return_time_stamps=False,
        )

    print_transcription_results(results)
    print(f"\n推理耗时: {t.elapsed:.3f} 秒")

    return results, t.elapsed


def print_npu_memory(device_props):
    """打印 NPU 内存使用情况，返回已分配内存（GB）。"""
    print_section_header(5, "NPU 内存使用情况")

    mem_allocated = bytes_to_gb(torch_npu.npu.memory_allocated(DEVICE_INDEX))
    mem_reserved = bytes_to_gb(torch_npu.npu.memory_reserved(DEVICE_INDEX))
    mem_total = bytes_to_gb(device_props.total_memory)

    print(f"已分配内存: {mem_allocated:.2f} GB")
    print(f"已预留内存: {mem_reserved:.2f} GB")
    print(f"总内存:     {mem_total:.2f} GB")
    print(f"内存使用率: {mem_allocated / mem_total * 100:.2f}%")

    return mem_allocated


def print_summary(load_time, infer_time, mem_allocated):
    """打印演示总结。"""
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("所有测试均成功完成！\n")
    print("总结:")
    print(f"  - Qwen3-ASR-1.7B 模型已成功适配到昇腾 NPU")
    print(f"  - 支持自动语言检测和指定语言识别")
    print(f"  - 模型加载时间: {load_time:.2f} 秒")
    print(f"  - 单次推理时间: {infer_time:.3f} 秒")
    print(f"  - NPU 内存使用: {mem_allocated:.2f} GB")
    print("=" * SEPARATOR_WIDTH)


# -- 主流程 ------------------------------------------------------------------

def main():
    print("=" * SEPARATOR_WIDTH)
    print("昇腾 NPU Qwen3-ASR-1.7B 语音识别演示")
    print("=" * SEPARATOR_WIDTH)

    device_props = check_npu_environment()
    model, load_time = load_model()

    # 自动检测语言
    print_section_header(3, "单个音频识别（自动检测语言）")
    _, infer_time = transcribe_audio(model, AUDIO_PATH, language=None)

    # 指定中文
    print_section_header(4, "指定语言识别（强制中文）")
    transcribe_audio(model, AUDIO_PATH, language="Chinese")

    mem_allocated = print_npu_memory(device_props)
    print_summary(load_time, infer_time, mem_allocated)


if __name__ == "__main__":
    main()
