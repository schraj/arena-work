# ARENA on a Runpod GPU pod

Local macOS (`.venv`, MPS) covers chapters 0, 1, 3 and most of 2. This pod exists for the
four things Apple Silicon cannot do:

| Blocked on macOS | Needs |
|---|---|
| `jax[cuda12]==0.10.1` + `brax==0.14.2` — [2.3] GPU MuJoCo PPO | CUDA 12 |
| `envpool==1.2.5` — Atari bonus vectorized envs | Linux x86_64 wheels only |
| `bitsandbytes` — quantization | Linux/CUDA wheels |
| MuJoCo offscreen video rendering | EGL (NVIDIA driver) or OSMesa |

## Runbook

```bash
./runpod/provision.sh                       # local: volume + pod + ssh config block (~$0.74/hr)
# paste the printed block into ~/.ssh/config
scp -P <port> -i <key> runpod/pod-*.sh root@<ip>:/workspace/
ssh arena-pod 'bash /workspace/pod-install.sh'     # one-time, ~15 min, ends with a 6-point verify
```

Override defaults via env: `GPU=... DC=... VOL_GB=... STOP_AFTER=... ./runpod/provision.sh`.

### After every stop/start
```bash
runpodctl pod start <pod-id>
runpodctl ssh info <pod-id>                 # the external TCP PORT CHANGES — update ~/.ssh/config
ssh arena-pod 'bash /workspace/pod-bootstrap.sh'   # reinstalls apt packages, rewrites env.sh
```

### Teardown
```bash
runpodctl pod stop   <pod-id>    # keeps pod + volume; you still pay for disk (~$7/mo at 100GB)
runpodctl pod remove <pod-id>    # deletes the pod; volume survives
runpodctl network-volume delete <vol-id>   # pod must be removed first
runpodctl pod list && runpodctl network-volume list
```

## Why not just run the repo's `install.sh`

It targets a throwaway RunPod box: Miniconda into `~/miniconda3`, repo cloned into `~`,
`apt` packages installed at top level. **Everything outside `/workspace` is wiped on every
pod stop** — you would rebuild a ~30GB conda env each session. `pod-install.sh` does the same
steps rooted at `/workspace`; `pod-bootstrap.sh` re-does only the genuinely ephemeral parts.

## Gotchas (from golden path 06, live-verified)

- **SSH keys are injected at boot.** Register before creating the pod, not after.
- **`Running` ≠ sshd up.** After a start, SSH can refuse connections for ~90s.
- **`ssh info` can return a stale port** immediately after `start` — re-read it if refused.
- **Volume is DC-locked.** The pod must be created in the same data center as its volume.
- **Only `/workspace` persists.** Keep code, envs, caches, and wandb runs there.
- **GPU choice:** RTX 4090 (sm_89) is deliberate — `jax==0.10.1`/`brax==0.14.2` are pinned
  versions predating solid Blackwell (sm_120) support, so avoid 5090 / RTX PRO cards here.
