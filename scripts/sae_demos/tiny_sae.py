"""Minimal SAE trained on gpt2-small layer-7 activations: shows the L1 / sparsity trade-off."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'chapter1_transformer_interp/exercises'))
import torch as t, einops
from datasets import load_dataset
from sae_lens import HookedSAETransformer

device = t.device("mps")
t.manual_seed(0)
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
HOOK, D_IN = "blocks.7.hook_resid_pre", 768

# ---- 1. collect activations (this is the SAE's training data: vectors, not text) ----
ds = load_dataset("roneneldan/TinyStories", split="train[:4000]")
texts = [x for x in ds["text"] if len(x) > 400][:1024]
acts = []
for i in range(0, len(texts), 32):
    tok = gpt2.to_tokens(texts[i:i+32], prepend_bos=True)[:, :128]
    _, cache = gpt2.run_with_cache(tok, names_filter=[HOOK], stop_at_layer=8)
    acts.append(einops.rearrange(cache[HOOK][:, 1:], "b s d -> (b s) d"))   # drop BOS
X = t.cat(acts); del acts
scale = D_IN**0.5 / X.norm(dim=-1).mean()
X = X * scale                      # SAELens convention: mean act norm -> sqrt(d_in)
print(f"activation dataset: {tuple(X.shape)}, mean norm {X.norm(dim=-1).mean():.1f}\n")

# ---- 2. the SAE itself: two matrices and two biases ----
class SAE(t.nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()
        W = t.nn.init.kaiming_uniform_(t.empty(d_sae, d_in, device=device))
        W = W / W.norm(dim=-1, keepdim=True)          # unit-norm decoder rows
        self.W_dec = t.nn.Parameter(W.clone())
        self.W_enc = t.nn.Parameter(W.clone().T)      # tied init (not tied weights)
        self.b_enc = t.nn.Parameter(t.zeros(d_sae, device=device))
        self.b_dec = t.nn.Parameter(t.zeros(d_in, device=device))

    def forward(self, x):
        f = t.relu((x - self.b_dec) @ self.W_enc + self.b_enc)   # encode: sparse, overcomplete
        return f, f @ self.W_dec + self.b_dec                    # decode: back to d_in

D_SAE, STEPS, BATCH = D_IN * 8, 600, 4096
print(f"{'l1_coeff':>9} | {'L0':>6} | {'expl.var':>8} | {'dead':>5} | recon_loss  sparsity_loss")
for l1 in (0.0, 0.3, 1.0, 4.0):
    sae, opt = SAE(D_IN, D_SAE), None
    opt = t.optim.Adam(sae.parameters(), lr=1e-3)
    fired = t.zeros(D_SAE, dtype=t.bool, device=device)
    for step in range(STEPS):
        x = X[t.randint(0, len(X), (BATCH,), device=device)]
        f, xhat = sae(x)
        recon = (xhat - x).pow(2).sum(-1).mean()
        sparsity = f.abs().sum(-1).mean()
        warm = min(1.0, step / (STEPS * 0.1))              # L1 warm-up avoids early dead latents
        (recon + l1 * warm * sparsity).backward()
        with t.no_grad():   # keep decoder rows unit-norm so activations stay comparable
            sae.W_dec.data /= sae.W_dec.data.norm(dim=-1, keepdim=True)
        opt.step(); opt.zero_grad()
        if step > STEPS - 50: fired |= (f > 0).any(0)
    with t.no_grad():
        f, xhat = sae(X[:8192])
        ev = 1 - ((xhat - X[:8192]).var() / X[:8192].var()).item()
        print(f"{l1:9.1f} | {(f > 0).float().sum(-1).mean():6.1f} | {ev:8.3f} | "
              f"{(~fired).sum().item():5d} | {recon.item():10.2f}  {sparsity.item():13.1f}")
