"""
Check every dataset's rating files against its committed manifest.

    python experiments/verify_data.py

Run it on the laptop before transferring, and on worker1 after. Standard
library only, so it runs before torch is even installed.

The manifests must reach worker1 through git and the rating files
through scp -- never both through scp. A manifest copied alongside the
data it describes proves only that the two copies agree with each other;
the point is to prove the data matches what was committed. That is why
the second check below exists: if scp overwrote a manifest with a
different one, `git diff` shows it.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path("data")


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def manifests_match_git():
    """True if every manifest on disk is identical to the committed one."""
    try:
        out = subprocess.run(
            # --untracked-files=all: without it git collapses a directory of
            # untracked files into one "?? data/" line, and an uncommitted
            # manifest -- the exact state this is meant to catch -- would
            # slip past a per-file filter.
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "data/"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  ! could not run git: {exc}")
        return False

    changed = [line for line in out.splitlines() if line.endswith("manifest.json")]
    if changed:
        print("  ! manifests differ from what is committed:")
        for line in changed:
            print(f"      {line}")
        print("    '??' or 'A ' means not committed yet: commit on the laptop, push,")
        print("           and pull on worker1 before transferring the rating files.")
        print("    ' M' means modified: this copy does not match the committed one.")
        return False
    return True


def main():
    manifests = sorted(DATA_ROOT.glob("*/manifest.json"))
    if not manifests:
        print(f"no manifests under {DATA_ROOT}/ -- run from the repo root")
        return 1

    failures = 0
    print(f"{'dataset':<14} {'file':<14} {'bytes':>12}  result")
    print("-" * 52)

    for manifest_path in manifests:
        dataset = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text())

        for name, expected in sorted(manifest.get("files", {}).items()):
            path = manifest_path.parent / name
            if not path.is_file():
                print(f"{dataset:<14} {name:<14} {'--':>12}  MISSING")
                failures += 1
                continue

            size = path.stat().st_size
            if size != expected["bytes"]:
                print(f"{dataset:<14} {name:<14} {size:>12,}  WRONG SIZE "
                      f"(manifest says {expected['bytes']:,})")
                failures += 1
                continue

            if sha256(path) != expected["sha256"]:
                print(f"{dataset:<14} {name:<14} {size:>12,}  CHECKSUM MISMATCH")
                failures += 1
                continue

            print(f"{dataset:<14} {name:<14} {size:>12,}  ok")

    print()
    git_ok = manifests_match_git()
    if git_ok:
        print("  manifests match the committed versions")

    if failures or not git_ok:
        print(f"\nFAILED: {failures} file problem(s)"
              f"{'' if git_ok else ', manifests not as committed'}")
        return 1

    print(f"\nall {len(manifests)} datasets verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())