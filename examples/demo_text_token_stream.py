import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


MODEL_PATH = "/sn-post/kckczhang/checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT_PATH = "/sn-post/kckczhang/Qwen3-TTS-streaming/examples/text_token_stream.wav"
TEXT = "Hi, I am Qwen."


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
    emit_every_frames=8,
    decode_window_frames=80,
    use_optimized_decode=True,
)

chunks = []
session.start(text_ids[0])
for token_id in text_ids[1:]:
    result = session.step(token_id)
    if result is not None:
        chunks.append(result)

for _ in range(160):
    result = session.step(text_finished=not session.text_finished)
    if result is not None:
        chunks.append(result)
    if session.codec_finished:
        break

final_chunk = session.flush()
if final_chunk is not None:
    chunks.append(final_chunk)

if not chunks:
    raise RuntimeError("The session produced no PCM")

sample_rate = chunks[0][1]
audio = __import__("numpy").concatenate([chunk for chunk, _ in chunks])
sf.write(OUTPUT_PATH, audio, sample_rate)
print(f"wrote={OUTPUT_PATH}")
print(f"seconds={len(audio) / sample_rate:.3f}")
print(f"frames={len(session.codes)}")
