#!/usr/bin/env bash
# One-time setup for the talking-character feature (SadTalker + face enhancer).
# Downloads about 1.5 GB. Safe to run again (skips what's already there).
cd "$(dirname "$0")"

if [ ! -d SadTalker ]; then
  git clone --depth 1 https://github.com/OpenTalker/SadTalker.git || { echo "❌ Could not download SadTalker"; exit 1; }
fi

echo "Installing packages..."
pip install -q face_alignment kornia yacs pydub resampy safetensors av \
  librosa scikit-image joblib scipy tqdm pyyaml

# basicsr is old and sometimes fails to build on new Python. Try a few ways.
if ! python -c "import basicsr" 2>/dev/null; then
  pip install -q basicsr \
  || pip install -q --no-build-isolation basicsr \
  || pip install -q basicsr-fixed \
  || echo "⚠️ basicsr could not be installed – face sharpening will be turned off."
fi
# Install these without letting them re-install basicsr
pip install -q --no-deps facexlib gfpgan || echo "⚠️ gfpgan not installed – face sharpening will be off."
pip install -q filterpy lmdb addict tb-nightly 2>/dev/null || true

# Make the 2023 code work with today's Python/NumPy
python patch_sadtalker.py SadTalker

mkdir -p SadTalker/checkpoints SadTalker/gfpgan/weights
dl () { [ -s "$2" ] || wget -q --show-progress "$1" -O "$2" || echo "⚠️ download failed: $1"; }

R=https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc
dl $R/mapping_00109-model.pth.tar       SadTalker/checkpoints/mapping_00109-model.pth.tar
dl $R/mapping_00229-model.pth.tar       SadTalker/checkpoints/mapping_00229-model.pth.tar
dl $R/SadTalker_V0.0.2_256.safetensors  SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors
dl $R/SadTalker_V0.0.2_512.safetensors  SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors

W=SadTalker/gfpgan/weights
dl https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth     $W/alignment_WFLW_4HG.pth
dl https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth $W/detection_Resnet50_Final.pth
dl https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth           $W/GFPGANv1.4.pth
dl https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth       $W/parsing_parsenet.pth

# --- Lip-sync for moving (cinematic) videos: Wav2Lip ---
[ -d Wav2Lip ] || git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git
mkdir -p Wav2Lip/checkpoints
# sharper "GAN" version first, plain version as backup
dl https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth Wav2Lip/checkpoints/wav2lip_gan.pth
dl https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth       Wav2Lip/checkpoints/wav2lip.pth

echo "Checking setup..."
python - <<'EOF'
ok = True
for mod in ["facexlib", "kornia", "safetensors", "librosa"]:
    try:
        __import__(mod)
    except Exception as e:
        ok = False
        print(f"❌ {mod} failed to load: {e}")
try:
    import gfpgan  # noqa
    print("✅ Face sharpening available")
except Exception as e:
    print(f"⚠️ Face sharpening off ({e.__class__.__name__}) – talking videos still work")
print("✅ Talking-character setup complete." if ok else "❌ Setup has problems – send a screenshot of this output.")
EOF
