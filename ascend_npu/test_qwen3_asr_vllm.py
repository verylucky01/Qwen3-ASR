import os
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

import librosa
import numpy as np

audio_path = "/root/demo3/ascend_audio.mp3"
audio_data, sr = librosa.load(audio_path, sr=16000)
print(f"Audio loaded: {audio_data.shape}, sr={sr}, duration={len(audio_data)/sr:.2f}s")

from vllm import LLM, SamplingParams

model_path = "/root/demo4/Qwen3-ASR-1.7B"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    enforce_eager=True,
    limit_mm_per_prompt={"audio": 1},
)

prompt = {
    "prompt": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n",
    "multi_modal_data": {
        "audio": (audio_data, sr),
    },
}

sampling_params = SamplingParams(
    max_tokens=200,
    temperature=0,
    stop=["<|im_end|>", "<|endoftext|>"],
)
outputs = llm.generate([prompt], sampling_params=sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    print(f"\nGenerated {len(token_ids)} tokens")
    print(f"Token IDs: {list(token_ids[:20])}...")
    print("\n" + "="*60)
    print("ASR Output:", generated_text)
    print("="*60)
