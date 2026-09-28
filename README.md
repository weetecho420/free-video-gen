# 🎬 Free AI Video Studio

Make short AI videos **for free** — with **English and Tagalog voices**.

| Tab | What you give it | What you get |
|---|---|---|
| 🗣️ **Talking character** | A picture with a face + a script | The character **speaks your script** with lip-sync |
| 🎙️ **Video + voice-over** | A picture + motion prompt + script | A **moving video** with a narrator voice |
| 🎬 **Silent video** | A picture + motion prompt | A ~10-second moving video |

Runs on Google Colab's **free GPU**. No API keys, no subscriptions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weetecho420/free-video-gen/blob/main/Free_Video_Generator.ipynb)

## How to use

1. Click **Open In Colab**.
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save.
3. `Runtime` → **Run all** (setup takes about 5–8 minutes).
4. Click the `https://xxxx.gradio.live` link at the bottom.

## Voices

| Voice | Language |
|---|---|
| Blessica (female), Angelo (male) | Tagalog / Filipino |
| Rosa (female), James (male) | English, Filipino accent |
| Jenny, Aria (female), Guy, Andrew (male) | English (US) |

You can also **upload or record your own voice** in the Talking character tab.

## Tips

- **Script length:** about 25 words ≈ 10 seconds of speech.
- **Talking character picture:** one person, face clearly visible, looking at the camera. Works with photos and realistic/cartoon drawings that have a normal human face.
- **Motion prompts:** describe movement and camera — *"slow zoom in, hair blowing in the wind"*. English prompts work best (the voice can still be Tagalog).
- Out of memory? Lower resolution / FPS in Settings, or use `Runtime → Restart session` between different tabs.

## How long it takes (free T4 GPU)

- Talking character (10s): ~1–3 minutes
- Moving video (10s): ~5–15 minutes (first time also downloads ~20 GB)

## What's inside

- `app.py` — the web page (3 tabs)
- `voice.py` — free text-to-speech (edge-tts, with gTTS as backup)
- `talking.py` — lip-sync (SadTalker) + voice-over merge
- `generate.py` — image-to-video (LTX-Video)
- `setup_talking.sh`, `patch_sadtalker.py` — installs SadTalker and fixes it for today's Python
- `Free_Video_Generator.ipynb` — the Colab notebook (start here)

## Credits & licenses

[LTX-Video](https://github.com/Lightricks/LTX-Video) · [SadTalker](https://github.com/OpenTalker/SadTalker) (Apache-2.0) · [GFPGAN](https://github.com/TencentARC/GFPGAN) · [edge-tts](https://github.com/rany2/edge-tts).
Check each model's license before commercial use. Only use pictures and voices you have permission to use.
