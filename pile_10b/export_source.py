"""Export source only; never run git add on the experiment workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output.resolve()
    if target == root or root in target.parents:
        raise ValueError("Export outside the experiment workspace")
    if target.exists():
        raise ValueError("Output must not exist, to avoid mixing old files or credentials")
    skip = {"data", "results", "runs", "__pycache__", "node_modules"}
    secret = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
    accepted = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            continue
        if any(part.startswith((".", "cache", "token_cache", "archive")) or part in skip for part in rel.parts[:-1]):
            continue
        if path.name.startswith(".") and rel.as_posix() != ".gitignore":
            continue
        if not (path.suffix in {".py", ".sh", ".tex", ".md"} or
                path.name in {"requirements.txt", ".gitignore"} or rel.as_posix() == "pile_10b/config.json"):
            continue
        if path.stat().st_size > 10 * 2**20:
            raise ValueError("Unexpectedly large source: " + str(rel))
        content = path.read_text(encoding="utf-8")
        if secret.search(content):
            raise ValueError("Possible credential in " + str(rel) + "; export stopped")
        accepted.append((path, rel))
    target.mkdir(parents=True)
    manifest = []
    for path, rel in accepted:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest.append(dict(path=str(rel), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (target / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported %d source/document files to %s" % (len(manifest), target))
    print("Review before publishing; automated credential detection is not exhaustive.")


if __name__ == "__main__":
    main()
