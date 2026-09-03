#!/usr/bin/env python3
"""Block secrets from reaching a public repo.

Notebooks are the real hazard: a cell that printed a client object, an `hf auth login`
confirmation, or an os.environ dump keeps that text in the saved OUTPUT long after the
source looks clean. So this scans notebook outputs, not just source.
"""
import json, re, sys, pathlib

PATTERNS = [
    ("OpenAI key",     re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{20,}")),
    ("Anthropic key",  re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("HuggingFace",    re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("AWS key id",     re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("wandb key",      re.compile(r"wandb(?:\.login\(key=|\s+login\s+)['\"]?([0-9a-f]{40})")),
    ("assigned secret", re.compile(
        r"(?i)\b(api[_-]?key|auth[_-]?token|access[_-]?token|secret|password)\b\s*[=:]\s*"
        r"['\"][A-Za-z0-9_\-]{16,}['\"]")),
]
# A vendor-prefixed key (sk-…, hf_…, AKIA…) is unambiguous: never suppress it on context.
# Only the generic "name = literal" rule needs a suppressor, and it is applied to the
# match's OWN LINE — a nearby innocent os.environ must not excuse a real key next to it.
HIGH_CONFIDENCE = {"OpenAI key", "Anthropic key", "HuggingFace", "AWS key id", "Google API key"}
SAFE = re.compile(r"(?i)(os\.environ|getenv|dotenv|<your|your[_-]?key|xxx+|\.\.\.|\$\{?[A-Z_]+)")
PLACEHOLDER = re.compile(r"(?i)(xxx+|your|example|placeholder|redacted|\.\.\.)")

def scan_text(text, where, hits):
    text = text or ""
    for name, pat in PATTERNS:
        for m in pat.finditer(text):
            if name in HIGH_CONFIDENCE:
                if PLACEHOLDER.search(m.group(0)):
                    continue
            else:
                start = text.rfind("\n", 0, m.start()) + 1
                end = text.find("\n", m.end())
                line = text[start:end if end != -1 else len(text)]
                if SAFE.search(line):
                    continue
            hits.append((where, name, m.group(0)[:12] + "…"))

def scan(path):
    hits = []
    p = pathlib.Path(path)
    if not p.is_file():
        return hits
    if p.name == ".env" or p.suffix in {".pem", ".key"}:
        hits.append((path, "credential file", "should never be committed"))
        return hits
    if p.suffix == ".ipynb":
        try:
            nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return hits
        for i, cell in enumerate(nb.get("cells", [])):
            scan_text("".join(cell.get("source", [])), f"{path}:cell{i}:source", hits)
            for out in cell.get("outputs", []):
                for field in ("text", "evalue"):
                    scan_text("".join(out.get(field) or []) if isinstance(out.get(field), list)
                              else (out.get(field) or ""), f"{path}:cell{i}:OUTPUT", hits)
                for mime, val in (out.get("data") or {}).items():
                    if mime.startswith("text/"):
                        scan_text("".join(val) if isinstance(val, list) else str(val),
                                  f"{path}:cell{i}:OUTPUT", hits)
    else:
        scan_text(p.read_text(encoding="utf-8", errors="replace"), path, hits)
    return hits

def main(argv):
    all_hits = [h for f in argv for h in scan(f)]
    if not all_hits:
        print(f"scan_secrets: clean ({len(argv)} file(s))")
        return 0
    print("scan_secrets: POTENTIAL SECRETS — not safe to publish", file=sys.stderr)
    for where, name, sample in all_hits:
        print(f"  {where}\n    {name}: {sample}", file=sys.stderr)
    print("\nRemove them (clear the notebook output cell), then retry.", file=sys.stderr)
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
