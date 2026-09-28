"""
Free Image + Prompt -> ~10-second video generator
Model: LTX-Video (Lightricks), open-source, runs on a FREE Google Colab / Kaggle T4 GPU.

Memory-safe loading for free Colab (≈12 GB RAM):
  1. The big text model (T5) is loaded straight onto the GPU, reads the prompt, then is deleted.
  2. The video model is then loaded straight onto the GPU.
So the computer's RAM never has to hold everything at once.

Usage:
  python generate.py --image photo.jpg --prompt "the woman smiles and waves at the camera"
"""
import argparse
import gc
import math

import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID = "Lightricks/LTX-Video"
MAX_TOKENS = 128

DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "deformed face, extra limbs, watermark, text"
)

_pipe = None
_tokenizer = None


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16  # T4 GPUs work best with float16
    return "cpu", torch.float32


def encode_prompts(prompt, negative_prompt):
    """Turn the text into numbers with the T5 model, then free it from memory."""
    global _tokenizer
    from transformers import T5EncoderModel, T5TokenizerFast

    device, dtype = _device_dtype()
    if _tokenizer is None:
        _tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")

    print("Reading the prompt (loading text model)...")
    enc = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map={"": device},
    )
    enc.eval()

    def one(text):
        t = _tokenizer(text, padding="max_length", max_length=MAX_TOKENS, truncation=True,
                       add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            emb = enc(t.input_ids.to(device))[0]
        return emb.to(dtype), t.attention_mask.bool().to(device)

    pe, pm = one(prompt)
    ne, nm = one(negative_prompt)
    del enc
    _free()
    return pe, pm, ne, nm


def get_pipe():
    """Load the video model once, straight onto the GPU (no text model inside)."""
    global _pipe
    if _pipe is None:
        device, dtype = _device_dtype()
        print("Loading video model (first time downloads ~10-20 GB)...")
        from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel

        # stream each big part straight onto the GPU so the computer's RAM stays low
        dm = {"": device} if device == "cuda" else None
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True, device_map=dm)
        _free()
        vae = AutoencoderKLLTXVideo.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True, device_map=dm)
        _free()
        _pipe = LTXImageToVideoPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, vae=vae, text_encoder=None, tokenizer=None,
            torch_dtype=dtype,
        )
        _pipe.to(device)
        _pipe.vae.enable_tiling()  # saves memory when decoding frames
        _free()
    return _pipe


def frames_for_seconds(seconds: float, fps: int) -> int:
    """LTX needs (frames - 1) divisible by 8. Round up so the clip is at least `seconds` long."""
    raw = int(seconds * fps)
    return math.ceil((raw - 1) / 8) * 8 + 1


def fit_size(image, max_side: int = 512):
    """Keep the image's shape, cap the long side, and snap both sides to multiples of 32."""
    w, h = image.size
    scale = max_side / max(w, h)
    w = max(32, int(w * scale) // 32 * 32)
    h = max(32, int(h * scale) // 32 * 32)
    return w, h


def generate(
    image_path,
    prompt,
    output="output.mp4",
    seconds=10,
    fps=16,
    steps=30,
    max_side=512,
    guidance=3.0,
    seed=42,
    negative_prompt=DEFAULT_NEGATIVE,
):
    # 1) text first (big model in, out), 2) then the video model
    pe, pm, ne, nm = encode_prompts(prompt, negative_prompt)
    pipe = get_pipe()

    image = load_image(image_path).convert("RGB")
    width, height = fit_size(image, max_side)
    image = image.resize((width, height))
    num_frames = frames_for_seconds(seconds, fps)

    print(f"Generating {num_frames} frames at {width}x{height}, {fps} fps "
          f"(~{num_frames / fps:.1f}s). This takes several minutes on a free T4...")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    frames = pipe(
        image=image,
        prompt_embeds=pe,
        prompt_attention_mask=pm,
        negative_prompt_embeds=ne,
        negative_prompt_attention_mask=nm,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).frames[0]
    _free()

    export_to_video(frames, output, fps=fps)
    print(f"Saved: {output}")
    return output


def generate_safe(image_path, prompt, output="output.mp4", seconds=10, fps=16, steps=30,
                  max_side=512, seed=42):
    """
    Same as generate(), but runs in a separate process. If the free GPU/RAM runs out,
    only that process dies – the web page keeps running and shows a clear message.
    """
    import os
    import subprocess
    import sys

    cmd = [sys.executable, os.path.abspath(__file__),
           "--image", str(image_path), "--prompt", prompt, "--output", str(output),
           "--seconds", str(seconds), "--fps", str(int(fps)), "--steps", str(int(steps)),
           "--max-side", str(int(max_side)), "--seed", str(int(seed))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode == 0 and os.path.exists(output):
        return output
    err = (proc.stderr or "")[-2500:]
    print(err)
    low = err.lower()
    if proc.returncode in (-9, 137) or "killed" in low:
        raise RuntimeError("OUT_OF_RAM")
    if "out of memory" in low or "outofmemory" in low:
        raise RuntimeError("OUT_OF_GPU")
    if "no space left" in low:
        raise RuntimeError("OUT_OF_DISK")
    raise RuntimeError("Video generation failed:\n" + err[-1200:])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Image + prompt -> video (free, LTX-Video)")
    p.add_argument("--image", required=True, help="Path or URL of the starting image")
    p.add_argument("--prompt", required=True, help="Describe the motion/scene")
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    generate(a.image, a.prompt, a.output, a.seconds, a.fps, a.steps, a.max_side, seed=a.seed)
