"""
Lip-sync a MOVING video to a voice (Wav2Lip), so a full-body cinematic clip can also talk.

make_lipsync_video(video_path, audio_path) -> mp4 with the mouth matching the voice.
Face detection uses facexlib's RetinaFace (weights already downloaded by setup_talking.sh).
"""
import os
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
WAV2LIP_DIR = pathlib.Path(os.environ.get("WAV2LIP_DIR", HERE / "Wav2Lip"))
_GAN = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
WAV2LIP_CKPT = _GAN if _GAN.exists() and _GAN.stat().st_size > 1e8 else WAV2LIP_DIR / "checkpoints" / "wav2lip.pth"
FACE_WEIGHTS_DIR = HERE / "SadTalker" / "gfpgan" / "weights"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 96
MEL_STEP = 16

_model = None
_detector = None


# ---------------- audio -> mel (same settings Wav2Lip was trained with) ----------------
def _melspectrogram(wav):
    import librosa
    from scipy import signal

    wav = signal.lfilter([1, -0.97], [1], wav)
    D = librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800)
    mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    S = np.dot(mel_basis, np.abs(D))
    min_level = np.exp(-100 / 20 * np.log(10))
    S = 20 * np.log10(np.maximum(min_level, S)) - 20
    return np.clip(8 * ((S + 100) / 100) - 4, -4, 4)


def _load_model():
    global _model
    if _model is None:
        sys.path.insert(0, str(WAV2LIP_DIR))
        from models import Wav2Lip  # from the Wav2Lip repo

        ckpt = torch.load(WAV2LIP_CKPT, map_location=DEVICE, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        m = Wav2Lip()
        m.load_state_dict(state)
        _model = m.to(DEVICE).eval()
    return _model


def _load_detector():
    global _detector
    if _detector is None:
        from facexlib.detection import init_detection_model
        _detector = init_detection_model(
            "retinaface_resnet50", half=False, device=DEVICE, model_rootpath=str(FACE_WEIGHTS_DIR)
        )
    return _detector


def _detect_boxes(frames, pad_bottom=15):
    """One face box per frame (largest face), smoothed so it doesn't jitter."""
    det = _load_detector()
    boxes, last = [], None
    for f in frames:
        with torch.no_grad():
            faces = det.detect_faces(f, 0.8)
        if faces is not None and len(faces):
            areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
            x1, y1, x2, y2 = faces[int(np.argmax(areas))][:4]
            h, w = f.shape[:2]
            box = [max(0, x1), max(0, y1), min(w, x2), min(h, y2 + pad_bottom)]
            last = box
        if last is None:
            boxes.append(None)
        else:
            boxes.append(list(last))
    if all(b is None for b in boxes):
        raise RuntimeError("No face found in the video. The face must be clearly visible and facing the camera.")
    first = next(b for b in boxes if b is not None)
    boxes = [b if b is not None else first for b in boxes]
    # smooth over 5 frames
    arr = np.array(boxes, dtype=np.float32)
    sm = np.copy(arr)
    for i in range(len(arr)):
        sm[i] = arr[max(0, i - 2): i + 3].mean(axis=0)
    return sm.astype(int)


def _read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError("Could not read the video.")
    return frames, fps


def _get_enhancer():
    """GFPGAN face sharpener (optional). Returns None if not available."""
    try:
        from gfpgan import GFPGANer
        w = FACE_WEIGHTS_DIR / "GFPGANv1.4.pth"
        if not w.exists():
            return None
        return GFPGANer(model_path=str(w), upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
    except Exception as e:
        print(f"Face sharpening off ({e.__class__.__name__})")
        return None


def make_lipsync_video(video_path, audio_path, out_path="lipsync.mp4", batch_size=32, sharpen=True):
    import librosa

    frames, fps = _read_video(video_path)
    wav, _ = librosa.load(str(audio_path), sr=16000)
    mel = _melspectrogram(wav)

    # cut the mel into one chunk per video frame
    mel_chunks, i, mult = [], 0, 80.0 / fps
    while True:
        start = int(i * mult)
        if start + MEL_STEP > mel.shape[1]:
            mel_chunks.append(mel[:, -MEL_STEP:])
            break
        mel_chunks.append(mel[:, start:start + MEL_STEP])
        i += 1

    # if the voice is longer than the video, play the video forward-then-backward to fill
    if len(frames) < len(mel_chunks):
        loop = frames + frames[::-1]
        frames = [loop[k % len(loop)] for k in range(len(mel_chunks))]
    frames = frames[: len(mel_chunks)]

    print(f"Lip-syncing {len(frames)} frames at {fps:.0f} fps...")
    boxes = _detect_boxes(frames)
    model = _load_model()
    enhancer = _get_enhancer() if sharpen else None

    tmp = tempfile.mkdtemp()
    silent = os.path.join(tmp, "silent.avi")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))

    for s in range(0, len(frames), batch_size):
        fr = frames[s:s + batch_size]
        bx = boxes[s:s + batch_size]
        mc = mel_chunks[s:s + batch_size]
        faces = [cv2.resize(f[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE)) for f, (x1, y1, x2, y2) in zip(fr, bx)]
        img = np.asarray(faces)
        masked = img.copy()
        masked[:, IMG_SIZE // 2:] = 0
        img_in = np.concatenate((masked, img), axis=3) / 255.0
        mel_in = np.asarray(mc)[..., np.newaxis]

        img_t = torch.FloatTensor(np.transpose(img_in, (0, 3, 1, 2))).to(DEVICE)
        mel_t = torch.FloatTensor(np.transpose(mel_in, (0, 3, 1, 2))).to(DEVICE)
        with torch.no_grad():
            pred = model(mel_t, img_t)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, (x1, y1, x2, y2) in zip(pred, fr, bx):
            f = f.copy()
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            # blend softly so the edges of the mouth patch don't show
            mask = np.zeros((y2 - y1, x2 - x1), np.float32)
            mh, mw = mask.shape
            cv2.ellipse(mask, (mw // 2, int(mh * 0.72)), (int(mw * 0.42), int(mh * 0.30)), 0, 0, 360, 1, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(1, mw // 25))[..., None]
            region = f[y1:y2, x1:x2].astype(np.float32)
            f[y1:y2, x1:x2] = (p * mask + region * (1 - mask)).astype(np.uint8)
            if enhancer is not None:
                try:
                    _, _, f = enhancer.enhance(f, has_aligned=False, only_center_face=True, paste_back=True)
                except Exception:
                    pass
            writer.write(f)
    writer.release()

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", str(audio_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest", str(out_path)],
        check=True,
    )
    print(f"Saved: {out_path}")
    return str(out_path)
