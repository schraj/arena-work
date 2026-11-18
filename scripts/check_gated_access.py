"""Check real file-level access to gated HF repos (metadata alone doesn't prove access)."""

import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

REPOS = sys.argv[1:] or ["google/gemma-2-2b", "google/gemma-2b-it"]

for repo in REPOS:
    try:
        hf_hub_download(repo, "config.json")
        print(f"OK       {repo}")
    except GatedRepoError:
        print(f"GATED    {repo}  -> accept the license at https://huggingface.co/{repo}")
    except Exception as e:
        print(f"ERROR    {repo}  -> {type(e).__name__}: {str(e).splitlines()[0][:100]}")
