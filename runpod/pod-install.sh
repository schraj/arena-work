#!/usr/bin/env bash
# Run this ON THE POD, once. Replaces the repo's install.sh, which puts Miniconda in
# ~/miniconda3 and the repo in ~ — both on EPHEMERAL disk that is wiped on every pod stop.
# Everything here goes on the network volume at /workspace instead.
set -euo pipefail

WS=/workspace
CONDA_DIR=$WS/miniconda3
ENV_NAME=arena-env
REPO_DIR=$WS/ARENA_materials

bash "$(dirname "$0")/pod-bootstrap.sh"        # apt packages (ephemeral — re-run after each start)

if [ ! -d "$CONDA_DIR" ]; then
  echo "==> Installing Miniconda to $CONDA_DIR (persistent)"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$CONDA_DIR" && rm /tmp/mc.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env list | grep -q "^$ENV_NAME " || conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"

[ -d "$REPO_DIR" ] || git clone https://github.com/ARENA-education/ARENA_materials.git "$REPO_DIR"

echo "==> Installing requirements.txt (the FULL Linux/CUDA set — this is the point of the pod)"
cd "$REPO_DIR"
pip install -U pip setuptools wheel
pip install -r requirements.txt
conda install -n "$ENV_NAME" ipykernel --update-deps --force-reinstall -y
python -m ipykernel install --name "$ENV_NAME" --display-name "ARENA (pod)"

echo "==> Verifying the four things macOS could not do"
python - <<'PY'
import traceback
def check(name, fn):
    try: print(f"  OK   {name}: {fn()}")
    except Exception as e: print(f"  FAIL {name}: {type(e).__name__}: {str(e)[:160]}")

check("torch CUDA",   lambda: __import__("torch").cuda.get_device_name(0))
check("jax GPU",      lambda: __import__("jax").devices())
check("brax",         lambda: __import__("brax").__version__)
check("envpool",      lambda: __import__("envpool").make_gymnasium("Breakout-v5", num_envs=2).observation_space.shape)
check("bitsandbytes", lambda: __import__("bitsandbytes").__version__)
def mujoco_render():
    import os, mujoco, numpy as np
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 3'/><body><geom size='.1'/><joint type='free'/></body></worldbody></mujoco>")
    r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d)
    return f"{r.render().shape} via MUJOCO_GL={os.environ.get('MUJOCO_GL','auto')}"
check("mujoco render", mujoco_render)
PY

cat <<EOF

=== Done ===
Every new shell needs:  source /workspace/env.sh
Jupyter (pod port 8888 is proxied):
  source /workspace/env.sh && jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser
  → https://<POD_ID>-8888.proxy.runpod.net
EOF
