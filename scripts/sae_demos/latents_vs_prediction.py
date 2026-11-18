"""One fixed dictionary, different inputs -> different subsets. And: does the logit lens predict?"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'chapter1_transformer_interp/exercises'))
import torch as t
from sae_lens import SAE, HookedSAETransformer

device = t.device("mps")
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.7.hook_resid_pre", device=str(device))
HOOK = "blocks.7.hook_resid_pre"

prompts = [
    "Mitigating the risk of extinction from AI should be a global",
    "She poured the milk into her morning cup of",
    "def add(a, b):\n    return a +",
]

print("SAME SAE, three inputs - which latents fire:\n")
active = []
for p in prompts:
    _, cache = gpt2.run_with_cache(p, names_filter=[HOOK], stop_at_layer=8)
    f = sae.encode(cache[HOOK])[0, -1]
    idx = set(f.nonzero().squeeze(-1).tolist()); active.append(idx)
    top = ", ".join(f"{i.item()}({a.item():.0f})" for a, i in zip(*f.topk(3)))
    print(f"  {p.splitlines()[-1]!r:<45} L0={len(idx):>3}  top: {top}")
print(f"\n  overlap between prompt 1 and 2: {len(active[0] & active[1])} latents"
      f" | 1 and 3: {len(active[0] & active[2])} latents")
print("  (the dictionary is identical every time - only the subset changes)")

# ---- does the logit lens from layer 7 predict the actual next token? ----
p = prompts[0]
logits_actual = gpt2(p, return_type="logits")[0, -1]
_, cache = gpt2.run_with_cache(p, names_filter=[HOOK], stop_at_layer=8)
f = sae.encode(cache[HOOK])[0, -1]
direct = (f @ sae.W_dec) @ gpt2.W_U          # direct path only: latents -> unembedding

print(f"\n\nPrompt: {p!r}")
print(f"  ACTUAL next-token prediction (full model, all 12 layers):")
print(f"    {[gpt2.to_string(i) for i in logits_actual.topk(5).indices]}")
print(f"  LOGIT LENS from the layer-7 latents alone (direct path):")
print(f"    {[gpt2.to_string(i) for i in direct.topk(5).indices]}")
overlap = len(set(logits_actual.topk(20).indices.tolist()) & set(direct.topk(20).indices.tolist()))
print(f"\n  overlap in top-20: {overlap}/20 tokens")
print(f"  ({12 - 7} more layers of attention + MLPs run after this SAE's hook point)")
