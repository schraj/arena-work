# ARENA — my work

My solutions to the [ARENA](https://github.com/ARENA-education/ARENA_materials) curriculum
(Alignment Research Engineer Accelerator) — chapters on ML fundamentals, transformer
interpretability, RL, LLM evaluations, and alignment science.

**This repo contains only the files I wrote.** The course material, tests, and official
solutions live in the [upstream repo](https://github.com/ARENA-education/ARENA_materials) and are
not reproduced here; paths below mirror upstream so each file drops back into a clone of it.
All course content is © Callum McDougall and the ARENA team.

## Layout

```
chapter0_fundamentals/exercises/...     # my completed exercise notebooks
chapter1_transformer_interp/exercises/...
chapter2_rl/exercises/...
chapter3_llm_evals/exercises/...
chapter4_alignment_science/exercises/...
```

## My tooling

Beyond the exercises, things I built to run the course on my own hardware:

| Path | What it does |
|---|---|
| `requirements-macos.txt` | Dependency set adapted for macOS arm64 / PyTorch MPS |
| `runpod/` | Provision + bootstrap a RunPod GPU pod for the CUDA-only sections |
| `scripts/reapply_notebook_cells.py` | Re-apply my notebook edits onto a fresh upstream sync, cell by cell (git can't merge notebook JSON) |
| `scripts/sae_demos/` | Small standalone experiments built while working through 1.3.3 (SAEs) |
| `scripts/check_gated_access.py`, `scripts/patch_torchvision_videoreader.py` | Environment fixes: HF gated-model access check, torchvision patch |

## How this repo is maintained

I work inside a clone of the upstream repo; `./sync.sh` copies across exactly the files
that differ from upstream — so this repo is my diff, by construction, and never drifts
into republishing course material.

```bash
./sync.sh          # ARENA_DIR=~/code/ARENA_materials, diffed against its `main` (upstream mirror)
git add -A && git commit -m "..."
```

A `pre-commit` hook runs `scripts/scan_secrets.py` over staged files, which scans notebook
**outputs** as well as source — a saved cell output is the easiest way to publish an API key
by accident.

## Environment

Developed on macOS (Apple Silicon, PyTorch MPS). The CUDA-only parts of chapter 2
(`jax`/`brax` GPU MuJoCo PPO, `envpool` Atari) were run on a Linux GPU box.
