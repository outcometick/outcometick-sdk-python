"""The free public sample archive, fetched once and cached.

One day of real archive per venue, the same files a subscriber downloads, laid
out exactly as the archive is. Published as GitHub Release assets (not the git
repository) by outcometick/polymarket-tick-data-samples and
outcometick/predict.fun-data-samples.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request

SAMPLES = {
    "polymarket": ("https://github.com/outcometick/polymarket-tick-data-samples/releases/latest/download/"
                   "polymarket-data-samples.tar.gz", "polymarket-data-samples"),
    "predict": ("https://github.com/outcometick/predict.fun-data-samples/releases/latest/download/"
                "predict-fun-data-samples.tar.gz", "predict-fun-data-samples"),
}


def _cache_root() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "outcometick", "samples")


def _safe_members(tar: tarfile.TarFile, top: str):
    """Regular files and directories under `top/`, nothing else.

    An archive is something downloaded from the internet: no absolute paths, no
    `..`, no links or devices, and nothing outside the one directory we expect.
    """
    for m in tar.getmembers():
        name = m.name
        parts = name.replace("\\", "/").split("/")
        if name.startswith(("/", "\\")) or ".." in parts or (len(name) > 1 and name[1] == ":"):
            raise ValueError(f"refusing archive member {name!r}")
        if parts[0] != top:
            raise ValueError(f"unexpected top-level entry {name!r} (want {top}/)")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing non-regular archive member {name!r}")
        yield m


def load_sample(venue: str = "polymarket", *, cache_dir: str | None = None, refresh: bool = False) -> str:
    """Download (once) and unpack the public sample for a venue; return its root.

    venue: "polymarket" or "predict". The archive is ~100-300 MB.
    """
    if venue not in SAMPLES:
        raise ValueError(f"unknown venue {venue!r}; one of {', '.join(SAMPLES)}")
    url, top = SAMPLES[venue]
    root = os.path.join(cache_dir or _cache_root(), venue)
    target = os.path.join(root, top)
    marker = os.path.join(root, ".complete")
    if not refresh and os.path.isfile(marker) and os.path.isdir(target):
        return target

    os.makedirs(root, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".download-", dir=root)
    try:
        tgz = os.path.join(work, "sample.tar.gz")
        req = urllib.request.Request(url, headers={"User-Agent": "outcometick-python"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tgz, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        unpacked = os.path.join(work, "x")
        os.makedirs(unpacked)
        with tarfile.open(tgz, "r:gz") as tar:
            members = list(_safe_members(tar, top))
            for m in members:
                dest = os.path.join(unpacked, *m.name.replace("\\", "/").split("/"))
                if m.isdir():
                    os.makedirs(dest, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                src = tar.extractfile(m)
                with src, open(dest, "wb") as fh:
                    shutil.copyfileobj(src, fh, length=1 << 20)
        if os.path.isdir(target):
            shutil.rmtree(target)
        if os.path.exists(marker):
            os.remove(marker)
        os.replace(os.path.join(unpacked, top), target)
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(url + "\n")
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)
