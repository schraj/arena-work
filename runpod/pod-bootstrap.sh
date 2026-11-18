#!/usr/bin/env bash
# Run this ON THE POD after EVERY start. A pod stop wipes everything outside /workspace:
# apt packages, ~/.bashrc, /root, /tmp. The conda env and repo on /workspace survive.
set -euo pipefail
WS=/workspace

echo "==> apt packages (wiped on every stop)"
apt-get update -qq
# libosmesa6: software GL fallback for MuJoCo. EGL (hardware) comes from the NVIDIA driver and
# is what mujoco_rendering.py probes first. ffmpeg/libgl round out video encoding + rendering.
apt-get install -y -qq git curl libosmesa6 libgl1 libglfw3 libegl1 ffmpeg >/dev/null

cat > $WS/env.sh <<'EOF'
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate arena-env
# Keep every cache on the persistent volume, or you re-download models after each stop.
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export PIP_CACHE_DIR=/workspace/.cache/pip
export WANDB_DIR=/workspace/wandb
# MUJOCO_GL is deliberately UNSET: part3_ppo/mujoco_rendering.py probes egl -> osmesa -> glfw
# itself and must do so at import time. Set it only to force one backend for debugging.
cd /workspace/ARENA_materials
EOF

grep -q 'workspace/env.sh' /root/.bashrc 2>/dev/null || echo 'source /workspace/env.sh' >> /root/.bashrc
echo "==> Bootstrap done. New shells auto-source /workspace/env.sh"
