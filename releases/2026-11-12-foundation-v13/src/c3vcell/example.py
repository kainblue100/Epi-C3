"""Fetch the small public Arc reference example with a pinned Git blob hash."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

SIZE = 4_991_092
BLOB_SHA1 = "be997430c2c6c37c605da0712e837ca096ccca7f"
COMMIT = "551849dc25354dff724edceb39a6108d997bf876"
URLS = (
    f"https://raw.githubusercontent.com/ArcInstitute/cell-eval2/{COMMIT}/docs/data/H1-VCC-2025-training.h5ad",
    f"https://github.com/ArcInstitute/cell-eval2/raw/{COMMIT}/docs/data/H1-VCC-2025-training.h5ad",
)


def fetch_example(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in URLS:
        fd, temp_name = tempfile.mkstemp(prefix=".c3vcell-example-", dir=output.parent)
        tmp = Path(temp_name)
        digest = hashlib.sha1(f"blob {SIZE}\0".encode())
        total = 0
        try:
            with os.fdopen(fd, "wb") as target, urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > SIZE:
                        raise ValueError(f"Example exceeds the expected {SIZE} bytes")
                    digest.update(chunk)
                    target.write(chunk)
            if total != SIZE or digest.hexdigest() != BLOB_SHA1:
                raise ValueError(
                    f"Example integrity check failed: size={total}, git_blob={digest.hexdigest()}"
                )
            os.link(tmp, output)
            print(f"Downloaded verified Arc example to {output}", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            failures.append(f"{url}: {exc}")
            print(f"Download failed: {failures[-1]}", flush=True)
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError("The public Arc example was unreachable:\n" + "\n".join(failures))
