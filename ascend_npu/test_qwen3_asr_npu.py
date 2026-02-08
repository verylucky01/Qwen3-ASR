"""Qwen3-ASR-1.7B Ascend NPU adaptation test script."""

import sys
import logging
import traceback
from pathlib import Path

import torch
import torch_npu
from qwen_asr import Qwen3ASRModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QWEN_ASR_ROOT = Path("/root/demo2/Qwen3-ASR")
MODEL_PATH = Path("/root/demo2/Qwen3-ASR-1.7B")
AUDIO_PATH = Path("/root/demo2/ascend_audio.mp3")

MODEL_DTYPE = torch.bfloat16
DEVICE = "npu:0"
MAX_INFERENCE_BATCH_SIZE = 32
MAX_NEW_TOKENS = 256

SEPARATOR = "=" * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_qwen_asr_importable() -> None:
    """Add the Qwen3-ASR project root to *sys.path* if not already present."""
    root_str = str(QWEN_ASR_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def check_npu_availability() -> None:
    """Verify that at least one Ascend NPU device is available.

    Raises:
        RuntimeError: If no NPU device is detected.
    """
    logger.info("Checking NPU devices ...")
    is_available = torch_npu.npu.is_available()
    device_count = torch_npu.npu.device_count()
    logger.info("  NPU available : %s", is_available)
    logger.info("  NPU count     : %d", device_count)

    if not is_available:
        raise RuntimeError("No Ascend NPU device detected")

    device_name = torch_npu.npu.get_device_name(0)
    logger.info("  NPU device    : %s", device_name)


def load_model() -> Qwen3ASRModel:
    """Load the Qwen3-ASR model onto the NPU.

    Returns:
        The loaded :class:`Qwen3ASRModel` instance.
    """
    logger.info("Loading Qwen3-ASR-1.7B model to NPU ...")
    model = Qwen3ASRModel.from_pretrained(
        str(MODEL_PATH),
        dtype=MODEL_DTYPE,
        device_map=DEVICE,
        max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    logger.info("  Model loaded successfully")
    return model


def run_transcription(model: Qwen3ASRModel) -> None:
    """Run speech-to-text on the configured audio file and log the results.

    Args:
        model: A loaded Qwen3-ASR model instance.
    """
    logger.info("Transcribing audio: %s", AUDIO_PATH)
    results = model.transcribe(
        audio=str(AUDIO_PATH),
        language=None,
        return_time_stamps=False,
    )
    logger.info("  Transcription succeeded")

    for idx, result in enumerate(results):
        logger.info("  Sample %d — language: %s, text: %s",
                     idx, result.language, result.text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: run all NPU adaptation checks and a transcription test."""
    ensure_qwen_asr_importable()

    logger.info(SEPARATOR)
    logger.info("Qwen3-ASR-1.7B Ascend NPU Adaptation Test")
    logger.info(SEPARATOR)

    check_npu_availability()
    model = load_model()
    run_transcription(model)

    logger.info(SEPARATOR)
    logger.info("All tests passed — Qwen3-ASR-1.7B is running on Ascend NPU")
    logger.info(SEPARATOR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Test failed:\n%s", traceback.format_exc())
        sys.exit(1)
