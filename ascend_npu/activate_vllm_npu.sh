# shellcheck shell=bash
# =============================================================================
# 文件名: activate_vllm_npu_new.sh
# 描述:   华为昇腾 NPU 环境激活脚本（vLLM 框架）
# 用法:   source activate_vllm_npu_new.sh
# 注意:   此脚本必须通过 source 命令加载，不能直接执行
# =============================================================================

# ---------------------------------------------------------------------------
# 区段 B: 防止直接执行
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "错误: 此脚本必须通过 source 命令加载，不能直接执行。" >&2
    echo "用法: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 区段 C: 可配置变量（可通过环境变量覆盖默认值）
# ---------------------------------------------------------------------------
_VLLM_CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
_VLLM_DRIVER_LIB_DIRS=(
    "${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64/driver"
    "${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/lib64/common"
)
_VLLM_CONDA_BASE="${CONDA_BASE:-/root/anaconda3}"
_VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-vllm_npu}"
_VLLM_SCRIPT_PATH="${BASH_SOURCE[0]}"

# ---------------------------------------------------------------------------
# 区段 D: 工具函数
# ---------------------------------------------------------------------------

_vllm_log_error() {
    echo "错误: $*" >&2
}

_vllm_log_info() {
    echo "$*"
}

# 检查冒号分隔的路径变量中是否已包含指定目录
_vllm_path_contains() {
    local path_var="$1"
    local dir="$2"
    case ":${path_var}:" in
        *":${dir}:"*) return 0 ;;
        *)            return 1 ;;
    esac
}

# 将目录追加到 LD_LIBRARY_PATH（幂等，不重复追加）
_vllm_add_to_ld_library_path() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        _vllm_log_error "驱动库路径不存在: $dir"
        return 1
    fi
    if ! _vllm_path_contains "${LD_LIBRARY_PATH:-}" "$dir"; then
        export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH:-}"
    fi
}

# ---------------------------------------------------------------------------
# 区段 E: 核心激活函数
# ---------------------------------------------------------------------------

# 加载 CANN 环境
_vllm_setup_cann() {
    if [[ ! -f "$_VLLM_CANN_ENV_SCRIPT" ]]; then
        _vllm_log_error "CANN 环境脚本不存在: $_VLLM_CANN_ENV_SCRIPT"
        return 1
    fi
    # shellcheck source=/dev/null
    source "$_VLLM_CANN_ENV_SCRIPT"
}

# 添加驱动库路径到 LD_LIBRARY_PATH
_vllm_setup_driver_paths() {
    local dir
    local has_error=0
    for dir in "${_VLLM_DRIVER_LIB_DIRS[@]}"; do
        if ! _vllm_add_to_ld_library_path "$dir"; then
            has_error=1
        fi
    done
    return "$has_error"
}

# 激活 conda 环境
_vllm_activate_conda() {
    local activate_script="${_VLLM_CONDA_BASE}/bin/activate"
    if [[ ! -f "$activate_script" ]]; then
        _vllm_log_error "Conda activate 脚本不存在: $activate_script"
        return 1
    fi
    # shellcheck source=/dev/null
    source "$activate_script" "$_VLLM_CONDA_ENV"
}

# ---------------------------------------------------------------------------
# 区段 F: 诊断展示函数
# ---------------------------------------------------------------------------

# 显示 torch / torch_npu 版本和 NPU 可用性
_vllm_show_torch_info() {
    python -c "
try:
    import torch
    import torch_npu
    print(f'  - torch: {torch.__version__}')
    print(f'  - torch_npu: {torch_npu.__version__}')
    print(f'  - NPU 可用: {torch_npu.npu.is_available()}')
except ImportError as e:
    print(f'  - 警告: 无法导入包: {e}')
" 2>/dev/null
}

# 显示已安装的相关包
_vllm_show_installed_packages() {
    pip list | grep -E 'torch|vllm|triton|transformers' | sed 's/^/  - /'
}

# 汇总显示诊断信息
_vllm_show_diagnostics() {
    local separator="========================================"

    _vllm_log_info "$separator"
    _vllm_log_info "vLLM NPU 环境已激活"
    _vllm_log_info "$separator"

    if command -v python &>/dev/null; then
        _vllm_log_info "Python 版本: $(python --version 2>&1)"
    else
        _vllm_log_error "Python 未找到"
    fi

    _vllm_log_info ""
    _vllm_log_info "已安装的包版本:"
    _vllm_show_torch_info
    _vllm_show_installed_packages
    _vllm_log_info ""
    _vllm_log_info "使用方法:"
    _vllm_log_info "  source $_VLLM_SCRIPT_PATH"
    _vllm_log_info "$separator"
}

# ---------------------------------------------------------------------------
# 区段 G: 主函数
# ---------------------------------------------------------------------------

_vllm_main() {
    # 第一步: 设置 CANN 环境
    if ! _vllm_setup_cann; then
        _vllm_log_error "CANN 环境设置失败，终止激活"
        return 1
    fi

    # 第二步: 添加驱动库路径（非致命错误，仅警告）
    if ! _vllm_setup_driver_paths; then
        _vllm_log_error "部分驱动库路径不存在，继续激活..."
    fi

    # 第三步: 激活 conda 环境
    if ! _vllm_activate_conda; then
        _vllm_log_error "Conda 环境激活失败，终止激活"
        return 1
    fi

    # 第四步: 显示诊断信息
    _vllm_show_diagnostics
}

_vllm_main

# ---------------------------------------------------------------------------
# 区段 H: 清理内部函数和变量，避免污染调用者 shell
# ---------------------------------------------------------------------------
unset -f _vllm_log_error _vllm_log_info _vllm_path_contains
unset -f _vllm_add_to_ld_library_path
unset -f _vllm_setup_cann _vllm_setup_driver_paths _vllm_activate_conda
unset -f _vllm_show_torch_info _vllm_show_installed_packages _vllm_show_diagnostics
unset -f _vllm_main
unset _VLLM_CANN_ENV_SCRIPT _VLLM_DRIVER_LIB_DIRS
unset _VLLM_CONDA_BASE _VLLM_CONDA_ENV _VLLM_SCRIPT_PATH
