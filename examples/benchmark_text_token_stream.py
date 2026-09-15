import time

import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


MODEL_PATH = "/sn-post/kckczhang/checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEXT = "Hi, I am Qwen. This is a streaming text token benchmark."


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    use_cuda_graphs=True,
    compile_mode="reduce-overhead",
    use_fast_codebook=False,
    compile_codebook_predictor=True,
)

ids = model._tokenize_texts([model._build_assistant_text(TEXT)])[0][0].tolist()
text_ids = ids[3:-5]
session = model.model.create_text_token_stream(
    language="English",
    speaker="Ryan",
    do_sample=False,
    subtalker_dosample=False,
    emit_every_frames=1,
    decode_window_frames=80,
    use_optimized_decode=True,
)

session.start(text_ids[0])
for token_id in text_ids[1:3]:
    session.step(token_id)

session = model.model.create_text_token_stream(
    language="English",
    speaker="Ryan",
    do_sample=False,
    subtalker_dosample=False,
    emit_every_frames=1,
    decode_window_frames=80,
    use_optimized_decode=True,
)

start = sync_time()
session.start(text_ids[0])
prefill_end = sync_time()
timings = []
pcm_samples = 0
first_pcm_end = None
for token_id in text_ids[1:]:
    step_start = sync_time()
    result = session.step(token_id)
    step_end = sync_time()
    timings.append(step_end - step_start)
    if result is not None:
        pcm_samples += len(result[0])
        if first_pcm_end is None:
            first_pcm_end = step_end

for _ in range(80):
    step_start = sync_time()
    result = session.step(text_finished=not session.text_finished)
    step_end = sync_time()
    timings.append(step_end - step_start)
    if result is not None:
        pcm_samples += len(result[0])
        if first_pcm_end is None:
            first_pcm_end = step_end
    if session.codec_finished:
        break

elapsed = sync_time() - start
audio_seconds = pcm_samples / 24000
print(f"text_tokens={len(text_ids)}")
print(f"prefill_ms={(prefill_end - start) * 1000:.1f}")
print(f"first_pcm_ms={(first_pcm_end - start) * 1000:.1f}")
print(f"steps={len(timings)}")
print(f"step_first_ms={timings[0] * 1000:.1f}")
if len(timings) > 2:
    steady = timings[2:]
    print(f"step_steady_mean_ms={sum(steady) / len(steady) * 1000:.1f}")
    print(f"step_steady_min_ms={min(steady) * 1000:.1f}")
print(f"pcm_seconds={audio_seconds:.3f}")
print(f"total_s={elapsed:.3f}")
print(f"rtf={elapsed / audio_seconds:.3f}" if audio_seconds else "rtf=nan")
