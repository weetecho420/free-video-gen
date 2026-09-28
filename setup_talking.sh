#!/usr/bin/env bash
# One-time setup for the talking-character feature (SadTalker + face enhancer).
# Downloads about 1.5 GB. Safe to run again (skips what's already there).
set -e
cd "$(dirname "$0")"

if [ ! -d SadTalker ]; then
  git clone --depth 1 https://github.com/OpenTalker/SadTalker.git
fi

pip install -q face_alignment kornia yacs pydub resampy safetensors basicsr facexlib gfpgan av \
  librosa scikit-image joblib scipy tqdm pyyaml

# Make the 2023 code work with today's Python/NumPy
python patch_sadtalker.py SadTalker

mkdir -p SadTalker/checkpoints SadTalker/gfpgan/weights
dl () { [ -s "$2" ] || wget -q --show-progress "$1" -O "$2"; }

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

echo "✅ Talking-character setup complete."
