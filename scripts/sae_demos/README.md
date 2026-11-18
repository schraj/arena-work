# SAE demo scripts (1.3.3)

Standalone scripts written while working through 1.3.3. Each runs on gpt2-small on MPS in
1-2 minutes and answers one question empirically. Run with the repo venv and HF_HOME set:

    HF_HOME="/Volumes/Jeremy SSD/huggingface" .venv/bin/python scripts/sae_demos/<script>.py

| script | question | result on 2026-09-18 |
|---|---|---|
| `tiny_sae.py` | what does the L1 coefficient do? | trains 4 tiny SAEs; L0 970 -> 8.9 and explained var 0.987 -> 0.718 as lambda goes 0 -> 4 |
| `dead_latents.py` | are dead latents unreachable or just rare? | 709/6144 never fire on any of 130k vectors; max pre-activation -0.00, median -0.09 |
| `what_is_latent.py` | can you name a latent from weights alone? | logit lens gives what it *promotes*; only data gives what *triggers* it |
| `latents_vs_prediction.py` | does the layer-7 logit lens predict the next token? | 0/20 overlap with the model's actual top-20 |
