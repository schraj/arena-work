"""Are 'dead' latents merely rare, or actually unreachable? Train one SAE and interrogate it."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'chapter1_transformer_interp/exercises'))
import torch as t, einops
from datasets import load_dataset
from sae_lens import HookedSAETransformer

device = t.device("mps"); t.manual_seed(0)
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
HOOK, D_IN = "blocks.7.hook_resid_pre", 768

ds = load_dataset("roneneldan/TinyStories", split="train[:4000]")
texts = [x for x in ds["text"] if len(x) > 400][:1024]
acts = []
for i in range(0, len(texts), 32):
    tok = gpt2.to_tokens(texts[i:i+32], prepend_bos=True)[:, :128]
    _, cache = gpt2.run_with_cache(tok, names_filter=[HOOK], stop_at_layer=8)
    acts.append(einops.rearrange(cache[HOOK][:, 1:], "b s d -> (b s) d"))
X = t.cat(acts) * (D_IN**0.5 / t.cat(acts).norm(dim=-1).mean()); del acts

class SAE(t.nn.Module):
    def __init__(self, d_in, d_sae):
        super().__init__()
        W = t.nn.init.kaiming_uniform_(t.empty(d_sae, d_in, device=device))
        W = W / W.norm(dim=-1, keepdim=True)
        self.W_dec = t.nn.Parameter(W.clone()); self.W_enc = t.nn.Parameter(W.clone().T)
        self.b_enc = t.nn.Parameter(t.zeros(d_sae, device=device))
        self.b_dec = t.nn.Parameter(t.zeros(d_in, device=device))
    def pre(self, x): return (x - self.b_dec) @ self.W_enc + self.b_enc
    def forward(self, x):
        f = t.relu(self.pre(x)); return f, f @ self.W_dec + self.b_dec

D_SAE, STEPS, BATCH, L1 = D_IN * 8, 600, 4096, 4.0
sae = SAE(D_IN, D_SAE); opt = t.optim.Adam(sae.parameters(), lr=1e-3)
for step in range(STEPS):
    x = X[t.randint(0, len(X), (BATCH,), device=device)]
    f, xhat = sae(x)
    ((xhat - x).pow(2).sum(-1).mean() + L1 * min(1.0, step/(STEPS*0.1)) * f.abs().sum(-1).mean()).backward()
    with t.no_grad(): sae.W_dec.data /= sae.W_dec.data.norm(dim=-1, keepdim=True)
    opt.step(); opt.zero_grad()

# --- interrogate every latent against the ENTIRE activation dataset, not a sample ---
with t.no_grad():
    max_pre = t.full((D_SAE,), -1e9, device=device)
    n_fire = t.zeros(D_SAE, device=device)
    for i in range(0, len(X), 16384):
        p = sae.pre(X[i:i+16384])
        max_pre = t.maximum(max_pre, p.max(0).values)
        n_fire += (p > 0).sum(0)

dead = n_fire == 0
print(f"\ndataset: {len(X):,} activation vectors, dictionary: {D_SAE} latents")
print(f"dead (never fire on ANY of the {len(X):,} vectors): {dead.sum().item()}")
print(f"\n                        max pre-activation over the whole dataset")
print(f"  dead latents        : max {max_pre[dead].max():.2f}, median {max_pre[dead].median():.2f}  (all < 0 -> ReLU always outputs 0)")
print(f"  live latents        : max {max_pre[~dead].max():.2f}, median {max_pre[~dead].median():.2f}")
print(f"\n  b_enc of dead       : median {sae.b_enc[dead].median():.2f}   (driven negative by the L1 penalty)")
print(f"  b_enc of live       : median {sae.b_enc[~dead].median():.2f}")
rare = (n_fire > 0) & (n_fire < len(X) * 0.0001)
print(f"\n  for contrast, {rare.sum().item()} latents are RARE but alive (<0.01% of tokens) - those still learn")
