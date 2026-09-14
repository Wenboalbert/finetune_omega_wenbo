import hashlib
import json
from pathlib import Path
import subprocess

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive writes: an existing artifact is never silently replaced.
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

def verify_provider(root, expected):
    root = Path(root).resolve()
    if git(root, "rev-parse", "HEAD") != expected or git(root, "status", "--porcelain"):
        raise RuntimeError("provider must be at the pinned clean commit")
    import vggt_omega
    if Path(vggt_omega.__file__).resolve().parent != root / "vggt_omega":
        raise RuntimeError("wrong provider imported: " + str(vggt_omega.__file__))
