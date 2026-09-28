"""
Talking character: picture + voice -> lip-synced video (SadTalker, free, open-source).
Also: add a voice-over track to any video.

Run setup_talking.sh once before using make_talking_video().
"""
import glob
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SADTALKER_DIR = pathlib.Path(os.environ.get("SADTALKER_DIR", HERE / "SadTalker"))


def make_talking_video(
    image_path,
    audio_path,
    out_dir="results",
    full_image=True,
    still=True,
    enhance_face=True,
    size=256,
    expression=1.0,
):
    """
    image_path   : photo or drawing with ONE clear, front-facing face
    audio_path   : the voice (wav/mp3)
    full_image   : True = keep the whole picture, False = zoom to face only
    still        : True = less head movement (looks more natural with full_image)
    enhance_face : sharpen the face with GFPGAN (slower, nicer)
    size         : 256 (faster) or 512 (sharper, slower)
    expression   : 0.5 calm ... 1.0 normal ... 2.0 very expressive
    """
    if not (SADTALKER_DIR / "inference.py").exists():
        raise RuntimeError("SadTalker is not installed yet. Run: bash setup_talking.sh")

    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", str(pathlib.Path(audio_path).resolve()),
        "--source_image", str(pathlib.Path(image_path).resolve()),
        "--result_dir", str(out_dir),
        "--preprocess", "full" if full_image else "crop",
        "--size", str(int(size)),
        "--expression_scale", str(float(expression)),
    ]
    if still:
        cmd.append("--still")
    if enhance_face:
        try:
            import gfpgan  # noqa: F401
        except Exception:
            print("Face sharpening not available – continuing without it.")
            enhance_face = False
    if enhance_face:
        cmd += ["--enhancer", "gfpgan"]

    print("Making talking video... (about 1-3 minutes per 10 seconds on a free T4)")
    proc = subprocess.run(cmd, cwd=SADTALKER_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-1500:]
        if "face" in tail.lower() and ("detect" in tail.lower() or "no face" in tail.lower()):
            raise RuntimeError("No face found. Use an image with one clear, front-facing face.")
        raise RuntimeError("Lip-sync failed:\n" + tail)

    videos = [v for v in glob.glob(str(out_dir / "**" / "*.mp4"), recursive=True)
              if os.path.getmtime(v) >= started - 1]
    if not videos:
        raise RuntimeError("Lip-sync finished but no video was found.\n" + proc.stdout[-1000:])
    result = max(videos, key=os.path.getmtime)
    print(f"Saved: {result}")
    return result


def add_voiceover(video_path, audio_path, out_path="video_with_voice.mp4"):
    """Put a narrator voice on top of a (silent) video. Video length stays the same."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_path), "-i", str(audio_path),
         "-filter_complex", "[1:a]apad[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
        check=True,
    )
    print(f"Saved: {out_path}")
    return out_path
