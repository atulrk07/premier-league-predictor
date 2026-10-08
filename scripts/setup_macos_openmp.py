"""Optional Apple Silicon fix when XGBoost cannot find libomp (no Homebrew needed).

Run with .venv/bin/python scripts/setup_macos_openmp.py after pip install.
Keeps the runtime and its license in .venv; rerun after recreating/moving the venv.
"""
import hashlib
import io
from pathlib import Path
import platform
import ssl
import subprocess
import sys
import tarfile
from urllib.request import urlopen

import certifi

URL = "https://repo.anaconda.com/pkgs/main/osx-arm64/llvm-openmp-22.1.2-h4714345_1.tar.bz2"
SHA256 = "74fcb92586ca84ed902f0b23b4e69072b2193d591491ffd1080be9eb80a129d2"


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This helper is for Apple Silicon macOS only")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this inside the project virtual environment")
    with urlopen(URL, context=ssl.create_default_context(cafile=certifi.where()), timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("OpenMP archive hash mismatch")
    destination = Path(sys.prefix) / "openmp"
    destination.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:bz2") as archive:
        archive.extractall(destination, filter="data")
    library = next(Path(sys.prefix).glob("lib/python*/site-packages/xgboost/lib/libxgboost.dylib"))
    rpath = str(destination / "lib")
    current = subprocess.check_output(["otool", "-l", str(library)], text=True)
    if f"path {rpath} " not in current:
        subprocess.run(["install_name_tool", "-add_rpath", rpath, str(library)], check=True)
    print("Local OpenMP runtime installed. Source:", URL)
    print("SHA256:", SHA256)


if __name__ == "__main__":
    main()
