#!/usr/bin/env python3
"""Re-apply my notebook edits onto fresh upstream notebooks, cell by cell.

Why this exists: upstream commits notebooks as single-line JSON and Jupyter re-saves
them multi-line, so `git merge` conflicts on every edited notebook. Instead we treat
each notebook as a list of cells and do a 3-way merge at cell granularity:

    BASE  = the upstream commit my edits were made against
    MINE  = my branch (edits + outputs)
    NEW   = the fresh upstream notebook in the working tree (what we write into)

For every cell I edited or added in MINE relative to BASE:
  * if upstream left that cell alone  -> take my source + outputs
  * if upstream ALSO edited that cell -> by default take mine (upstream edits are
    almost always typo/URL fixes), unless --prefer-upstream, and report it either way
  * cells I added (no counterpart)    -> insert after their preceding neighbour
Outputs from MINE are always carried over when the cell aligns.

Usage (from repo root, on the branch you want to update, after `git checkout NEW-upstream -- <notebooks>`
or after resetting main and switching to `work`):

    python scripts/reapply_notebook_cells.py --base <base-sha> --mine <branch-or-sha> [--dry-run] [--prefer-upstream] [notebooks...]

With no notebooks given it uses every .ipynb that differs between BASE and MINE.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys


def sh(*args: str) -> str:
    return subprocess.check_output(list(args)).decode()


def load_rev(rev: str, path: str) -> dict:
    return json.loads(sh("git", "show", f"{rev}:{path}"))


def src(cell: dict) -> str:
    return "".join(cell["source"])


def merge3(base: str, mine: str, theirs: str) -> tuple[str, bool]:
    """Line-level 3-way merge of one cell's source via `git merge-file`. Returns (text, clean)."""
    import os, subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, text in (("mine", mine), ("base", base), ("theirs", theirs)):
            fp = os.path.join(d, name)
            with open(fp, "w") as fh:
                fh.write(text if text.endswith("\n") else text + "\n")
            paths.append(fp)
        r = subprocess.run(["git", "merge-file", "-p", *paths], capture_output=True, text=True)
    out = r.stdout
    if not (mine.endswith("\n") or theirs.endswith("\n")):
        out = out[:-1] if out.endswith("\n") else out
    return out, r.returncode == 0


def align(a: list[str], b: list[str]) -> dict[int, int]:
    """Map index in a -> index in b for cells that correspond 1:1 (equal or same-size replace)."""
    m: dict[int, int] = {}
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "replace") and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
    return m


def reapply(path: str, base_rev: str, mine_rev: str, prefer_upstream: bool, dry_run: bool) -> None:
    base, mine = load_rev(base_rev, path), load_rev(mine_rev, path)
    with open(path) as fh:
        new = json.load(fh)
    bs, ms, ns = [src(c) for c in base["cells"]], [src(c) for c in mine["cells"]], [src(c) for c in new["cells"]]
    b2m, b2u = align(bs, ms), align(bs, ns)

    applied, both, unaligned, merged_clean, conflicts = [], [], [], [], []
    for i, j in b2m.items():
        mc = mine["cells"][j]
        i_edited = bs[i] != ms[j]
        if not i_edited and not mc.get("outputs"):
            continue
        if i not in b2u:
            unaligned.append(j)
            continue
        uc = new["cells"][b2u[i]]
        u_edited = ns[b2u[i]] != bs[i]
        if i_edited and u_edited:
            both.append(i)
            if prefer_upstream:
                continue
            merged, clean = merge3(bs[i], ms[j], ns[b2u[i]])
            if clean:  # both sides' line edits survive
                uc["source"] = merged.splitlines(keepends=True)
                merged_clean.append(i)
            else:      # true conflict: keep mine, flag for a manual look
                uc["source"] = mc["source"]
                conflicts.append(i)
        elif i_edited:
            uc["source"] = mc["source"]
        if mc.get("outputs"):
            uc["outputs"] = mc["outputs"]
            uc["execution_count"] = mc.get("execution_count")
        applied.append(i)

    # Cells I added: no base counterpart. Insert after nearest preceding cell that exists in NEW.
    added = [j for j in range(len(ms)) if j not in b2m.values() and ms[j].strip()]
    now_srcs = [src(c) for c in new["cells"]]
    inserted = []
    for j in added:
        if ms[j] in now_srcs:
            continue
        k = j - 1
        while k >= 0 and ms[k] not in now_srcs:
            k -= 1
        pos = now_srcs.index(ms[k]) + 1 if k >= 0 else 0
        new["cells"].insert(pos, mine["cells"][j])
        now_srcs.insert(pos, ms[j])
        inserted.append((j, pos))

    name = path.rsplit("/", 1)[-1]
    print(f"{name}: applied={len(applied)} inserted={len(inserted)} both-edited={both}"
          + (f" UNALIGNED(my cell idx)={unaligned}" if unaligned else ""))
    for i in both:
        d = [l for l in difflib.unified_diff(bs[i].splitlines(), ns[b2u[i]].splitlines(), lineterm="", n=0)
             if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        tag = ("kept UPSTREAM" if prefer_upstream else "MERGED both sides cleanly, upstream had"
               if i in merged_clean else "CONFLICT - took MINE, review; upstream had")
        print(f"    cell {i}: {tag} ({len(d)} changed lines)")
        for l in d[:4]:
            print("       ", l[:130])

    if not dry_run:
        with open(path, "w") as fh:
            json.dump(new, fh, indent=1, ensure_ascii=False)
            fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="upstream commit my edits were made against")
    ap.add_argument("--mine", required=True, help="branch/sha holding my edits and outputs")
    ap.add_argument("--prefer-upstream", action="store_true", help="on doubly-edited cells keep upstream's source")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("notebooks", nargs="*")
    a = ap.parse_args()

    nbs = a.notebooks or [
        l for l in sh("git", "diff", "--name-only", a.base, a.mine).splitlines() if l.endswith(".ipynb")
    ]
    if not nbs:
        print("no notebooks differ between base and mine")
        return 0
    for nb in nbs:
        reapply(nb, a.base, a.mine, a.prefer_upstream, a.dry_run)
    # sanity: every file still parses
    for nb in nbs:
        with open(nb) as fh:
            json.load(fh)
    print(f"{len(nbs)} notebooks {'checked' if a.dry_run else 'written'}, all parse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
