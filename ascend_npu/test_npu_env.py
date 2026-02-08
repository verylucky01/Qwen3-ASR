"""昇腾 NPU 环境测试脚本。

检测 PyTorch、torch_npu 版本信息，验证 NPU 设备可用性，
执行简单张量运算测试，并检查 vllm-ascend 是否可导入。
"""
import re
import sys
import subprocess
import torch
import torch_npu


SEPARATOR_WIDTH: int = 50


def print_section_header(number: int, title: str) -> None:
    """打印带编号的章节标题。"""
    print(f"\n{number}. {title}")


def check_pytorch_versions() -> bool:
    """打印 PyTorch 和 torch_npu 版本信息。"""
    print_section_header(1, "PyTorch 版本信息:")
    print(f"   torch 版本: {torch.__version__}")
    print(f"   torch_npu 版本: {torch_npu.__version__}")
    return True


def check_npu_devices() -> bool:
    """检测 NPU 设备可用性并枚举设备信息。

    Returns:
        有可用 NPU 设备时返回 True，否则返回 False。
    """
    print_section_header(2, "NPU 设备信息:")
    is_available: bool = torch_npu.npu.is_available()
    print(f"   NPU 是否可用: {is_available}")

    if not is_available:
        return False

    device_count: int = torch_npu.npu.device_count()
    print(f"   NPU 设备数量: {device_count}")
    for i in range(device_count):
        print(f"   NPU {i}: {torch_npu.npu.get_device_name(i)}")

    return True


def check_tensor_operations() -> bool:
    """在 NPU 上创建张量并验证基本运算。

    Returns:
        张量创建和运算成功返回 True，否则返回 False。
    """
    print_section_header(3, "简单张量操作测试:")
    try:
        x = torch.randn(3, 3).npu()
        y = torch.randn(3, 3).npu()
        z = x + y
        print("   ✓ 张量创建和运算成功")
        print(f"   设备: {z.device}")
        return True
    except RuntimeError as e:
        print(f"   ✗ 测试失败: {e}")
        return False


def check_vllm_ascend() -> bool:
    """尝试导入 vllm_ascend 并报告结果。

    Returns:
        导入成功返回 True，否则返回 False。
    """
    print_section_header(4, "vllm-ascend 信息: ")
    try:
        import vllm
        import vllm_ascend  # noqa: F401
        print("   ✓ vllm_ascend 导入成功")
        # 获取 pip list 输出
        pip_output = subprocess.run(
            ['pip', 'list'],
            capture_output=True,
            text=True
        ).stdout

        # 按行处理，过滤并添加前缀
        for line in pip_output.split('\n'):
            if re.search(r'vllm|triton|transformers', line, re.IGNORECASE):
                print(f"    - {line}")
        return True
    except ImportError as e:
        print(f"   ✗ vllm_ascend 导入失败: {e}")
        return False


def main() -> None:
    """运行所有 NPU 环境检查并以相应退出码退出。"""
    separator: str = "=" * SEPARATOR_WIDTH

    print(separator)
    print("昇腾 NPU 环境测试")
    print(separator)

    results: list[bool] = [
        check_pytorch_versions(),
        check_npu_devices(),
        check_tensor_operations(),
        check_vllm_ascend(),
    ]

    print(f"\n{separator}")
    print("环境测试完成！")
    print(separator)

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
