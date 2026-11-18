"""Two ways to ask what a latent means: from the weights alone, and from data."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'chapter1_transformer_interp/exercises'))
import torch as t, einops
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer

device = t.device("mps")
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.7.hook_resid_pre", device=str(device))
HOOK, LATENT = "blocks.7.hook_resid_pre", 17676

# ---------- (A) WEIGHTS ONLY: what does it *do* when it fires? ----------
# push the latent's decoder direction through the unembedding: the logit lens
logits = sae.W_dec[LATENT] @ gpt2.W_U
print(f"(A) from weights alone - tokens latent {LATENT} promotes when it fires:")
print("   ", [gpt2.to_string(i) for i in logits.topk(12).indices])
print("    tokens it suppresses:")
print("   ", [gpt2.to_string(i) for i in logits.topk(8, largest=False).indices])

# ---------- (B) DATA: what makes it fire? ----------
ds = load_dataset("NeelNanda/pile-10k", split="train[:1500]", streaming=False)
best = []   # (activation, decoded context, the token itself)
for i in range(0, 1500, 32):
    toks = gpt2.to_tokens(ds["text"][i:i+32], prepend_bos=True)[:, :128]
    _, cache = gpt2.run_with_cache(toks, names_filter=[HOOK], stop_at_layer=8)
    acts = sae.encode(cache[HOOK])[:, :, LATENT]          # (batch, seq)
    for b, s in zip(*(acts > 3).nonzero(as_tuple=True)):
        ctx = gpt2.to_string(toks[b, max(0, s-7):s+2])
        best.append((acts[b, s].item(), ctx.replace("\n", " "), gpt2.to_string(toks[b, s])))
best.sort(reverse=True)
print(f"\n(B) from data - top activating contexts over 1500 pile documents "
      f"({len(best)} firings above threshold):")
for act, ctx, tok in best[:10]:
    print(f"    {act:6.2f}  ...{ctx.strip()[-70:]!r}   (token fired on: {tok!r})")
