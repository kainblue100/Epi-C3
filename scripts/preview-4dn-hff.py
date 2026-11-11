"""Extract a small real HFFc6 Hi-C subset from the public 4DN object by HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

URL = (
    "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/"
    "bd06e6c5-765f-4660-8b1c-23b681bcfff0/4DNFIB59T7NN.mcool"
)
ETAG = "b2bb36baaba61e620940f758da9b8bc4-2723"
SIZE = 22_837_705_555
RESOLUTION = 1_000_000
CHROMOSOMES = ("chr21", "chr22")


class RangeFile(io.RawIOBase):
    def __init__(self, directory: Path) -> None:
        self.position = 0
        self.directory = directory
        self.page_size = 256 * 1024
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.transferred = 0
        request = urllib.request.Request(URL, method="HEAD", headers={"If-Match": f'"{ETAG}"'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                int(response.headers["Content-Length"]) != SIZE
                or response.headers.get("ETag", "").strip('"') != ETAG
                or response.headers.get("Accept-Ranges") != "bytes"
            ):
                raise RuntimeError("The published 4DN object metadata changed")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        )
        if position < 0:
            raise ValueError("Cannot seek before the start of the source object")
        self.position = position
        return position

    def readinto(self, buffer) -> int:
        remaining = min(len(buffer), max(0, SIZE - self.position))
        copied = 0
        while copied < remaining:
            page = self.position // self.page_size
            start = page * self.page_size
            if page not in self.cache:
                length = min(self.page_size, SIZE - start)
                if self.transferred + length > 512 * 1024**2:
                    raise OSError("Preview would exceed its 512 MiB range-transfer limit")
                if shutil.disk_usage(self.directory).free < 200 * 1024**3:
                    raise OSError("Preview stopped at the 200 GiB disk-space reserve")
                request = urllib.request.Request(URL, headers={
                    "Range": f"bytes={start}-{start + length - 1}", "If-Match": f'"{ETAG}"',
                })
                with urllib.request.urlopen(request, timeout=60) as response:
                    if (
                        response.status != 206
                        or response.headers.get("ETag", "").strip('"') != ETAG
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{start + length - 1}/{SIZE}"
                        or int(response.headers.get("Content-Length", "0")) != length
                    ):
                        raise RuntimeError("The source did not honor the version-matched object range")
                    chunk = response.read(length + 1)
                if len(chunk) != length:
                    raise RuntimeError("The source returned a short object range")
                self.transferred += length
                self.cache[page] = chunk
                if len(self.cache) > 64:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(page)
            data = self.cache[page]
            local = self.position - start
            count = min(remaining - copied, len(data) - local)
            buffer[copied:copied + count] = data[local:local + count]
            self.position += count
            copied += count
        return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    final = args.outdir / "HFFc6-chr21-chr22-1Mb.mcool"
    source_metadata = args.outdir / "source.json"
    remote = RangeFile(args.outdir)
    with h5py.File(remote, "r") as file:
        group = file[f"resolutions/{RESOLUTION}"]
        names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in group["chroms/name"][:]
        ]
        if any(name not in names for name in CHROMOSOMES):
            raise ValueError("Requested preview chromosomes are absent")
        selected_ids = sorted(names.index(name) for name in CHROMOSOMES)
        if [names[index] for index in selected_ids] != list(CHROMOSOMES):
            raise ValueError("Unexpected chromosome ordering in the source")
        lengths = np.asarray(group["chroms/length"][:], dtype=np.int64)
        offsets = np.asarray(group["indexes/chrom_offset"][:], dtype=np.int64)
        row_offsets = np.asarray(group["indexes/bin1_offset"][:], dtype=np.int64)
        bins_chrom = np.asarray(group["bins/chrom"][:], dtype=np.int64)
        starts = np.asarray(group["bins/start"][:], dtype=np.int64)
        ends = np.asarray(group["bins/end"][:], dtype=np.int64)
        weights = np.asarray(group["bins/weight"][:], dtype=np.float64)
        chosen = np.concatenate([
            np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
            for index in selected_ids
        ])
        mapping = np.full(len(bins_chrom), -1, dtype=np.int64)
        mapping[chosen] = np.arange(len(chosen))
        new_chrom = np.concatenate([
            np.full(offsets[index + 1] - offsets[index], position, dtype=np.int32)
            for position, index in enumerate(selected_ids)
        ])
        first_parts, second_parts, count_parts = [], [], []
        for index in selected_ids:
            start_row, end_row = row_offsets[offsets[index]], row_offsets[offsets[index + 1]]
            first = np.asarray(group["pixels/bin1_id"][start_row:end_row], dtype=np.int64)
            second = np.asarray(group["pixels/bin2_id"][start_row:end_row], dtype=np.int64)
            counts = np.asarray(group["pixels/count"][start_row:end_row], dtype=np.int32)
            keep = (bins_chrom[first] == index) & (bins_chrom[second] == index)
            first_parts.append(mapping[first[keep]])
            second_parts.append(mapping[second[keep]])
            count_parts.append(counts[keep])
        first = np.concatenate(first_parts)
        second = np.concatenate(second_parts)
        counts = np.concatenate(count_parts)
        if np.any(first < 0) or np.any(second < 0):
            raise RuntimeError("Preview bin remapping failed")
        with tempfile.NamedTemporaryFile(prefix=".4dn-preview-", suffix=".mcool", dir=args.outdir, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with h5py.File(temporary, "w") as out:
                out.attrs["format"] = "HDF5::MCOOL"
                out.attrs["format-version"] = 2
                view = out.create_group(f"resolutions/{RESOLUTION}")
                view.attrs["bin-size"] = RESOLUTION
                view.attrs["bin-type"] = "fixed"
                view.attrs["genome-assembly"] = "unknown"
                view.attrs["format"] = "HDF5::Cooler"
                view.attrs["format-version"] = 3
                view.attrs["storage-mode"] = "symmetric-upper"
                chroms = view.create_group("chroms")
                chroms.create_dataset("name", data=np.asarray(CHROMOSOMES, dtype="S64"))
                chroms.create_dataset("length", data=lengths[selected_ids])
                bins = view.create_group("bins")
                bins.create_dataset("chrom", data=new_chrom)
                bins.create_dataset("start", data=starts[chosen])
                bins.create_dataset("end", data=ends[chosen])
                bins.create_dataset("weight", data=weights[chosen])
                pixels = view.create_group("pixels")
                pixels.create_dataset("bin1_id", data=first, compression="gzip")
                pixels.create_dataset("bin2_id", data=second, compression="gzip")
                pixels.create_dataset("count", data=counts, compression="gzip")
                index = view.create_group("indexes")
                index.create_dataset(
                    "bin1_offset",
                    data=np.r_[0, np.cumsum(np.bincount(first, minlength=len(chosen)))],
                )
                sizes = [int(offsets[index + 1] - offsets[index]) for index in selected_ids]
                index.create_dataset("chrom_offset", data=np.r_[0, np.cumsum(sizes)])
            os.link(temporary, final)
        finally:
            temporary.unlink()
    sha = hashlib.sha256()
    with final.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    record = {
        "source_url": URL,
        "source_etag": ETAG,
        "source_accession": "4DNFIB59T7NN",
        "sha256": sha.hexdigest(),
        "input_scope": "derived chr21/chr22 cis pixels at 1 Mb from public HTTP ranges",
        "context": "HFFc6",
        "condition": "unknown",
        "assembly": "unknown",
        "assay": "Hi-C",
        "range_bytes_transferred": remote.transferred,
        "note": (
            "The original object's full checksum was not verified. HFFc6 is not HCT116; "
            "the treatment and assembly metadata are unknown."
        ),
    }
    with source_metadata.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(f"Saved {final} using {remote.transferred} range bytes")


if __name__ == "__main__":
    main()
