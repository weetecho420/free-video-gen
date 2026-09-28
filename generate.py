"""
Free Image + Prompt -> 10-second video generator
Model: LTX-Video (Lightricks), open-source, runs on a FREE Google Colab / Kaggle T4 GPU.

Usage:
  python generate.py --image photo.jpg --prompt "the woman smiles and waves at the camera"
"""
import argparse
import math

import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID = "Lightricks/LTX-Video"

DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "deformed face, extra limbs, watermark, text"
)

_pipe = None


def get_pipe():
    """Load the model once (about 20 GB download on the first run)."""
    global _pipe
    if _pipe is None:
        # T4 GPUs don't support bfloat16 well, so use float16
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _pipe = LTXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if torch.cuda.is_available():
            _pipe.enable_model_cpu_offload()  # keeps GPU memory under 15 GB
        _pipe.vae.enable_tiling()             # saves memory when decoding frames
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
    pipe = get_pipe()
    image = load_image(image_path)
    width, height = fit_size(image, max_side)
    image = image.resize((width, height))
    num_frames = frames_for_seconds(seconds, fps)

    print(f"Generating {num_frames} frames at {width}x{height}, {fps} fps "
          f"(~{num_frames / fps:.1f}s). This takes several minutes on a free T4...")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).frames[0]

    export_to_video(frames, output, fps=fps)
    print(f"Saved: {output}")
    return output


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
