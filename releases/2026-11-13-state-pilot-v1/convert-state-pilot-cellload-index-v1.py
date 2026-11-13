"""Create a verified CellLoad-compatible copy of frozen pilot inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np


INDEX_PATHS = ("obs/_index", "var/_index")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def values_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def read_nullable_index(node):
    if not isinstance(node, h5py.Group):
        raise ValueError(f"{node.name} is not a nullable string index")
    if node.attrs.get("encoding-type") != "nullable-string-array":
        raise ValueError(f"{node.name} has an unexpected encoding")
    if set(node.keys()) != {"values", "mask"} or np.any(node["mask"][:]):
        raise ValueError(f"{node.name} contains missing or unexpected index data")
    values = node["values"].asstr()[:].tolist()
    if not all(values) or len(values) != len(set(values)):
        raise ValueError(f"{node.name} must contain unique, nonempty strings")
    return values


def convert_index(file, path):
    values = read_nullable_index(file[path])
    parent, name = path.rsplit("/", 1)
    del file[path]
    dataset = file[parent].create_dataset(
        name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8")
    )
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"
    return values


def compare_attributes(left, right, path):
    if set(left.attrs) != set(right.attrs):
        raise ValueError(f"Attribute keys changed at {path}")
    for key in left.attrs:
        if not values_equal(left.attrs[key], right.attrs[key]):
            raise ValueError(f"Attribute {key} changed at {path}")


def compare_unchanged(left, right, path=""):
    compare_attributes(left, right, path or "/")
    if isinstance(left, h5py.Group):
        if not isinstance(right, h5py.Group) or set(left.keys()) != set(right.keys()):
            raise ValueError(f"Group structure changed at {path or '/'}")
        for key in left:
            child_path = f"{path}/{key}".lstrip("/")
            if child_path not in INDEX_PATHS:
                compare_unchanged(left[key], right[key], child_path)
        return
    if not isinstance(right, h5py.Dataset):
        raise ValueError(f"Dataset structure changed at {path}")
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"Dataset shape or type changed at {path}")
    if left.shape is None:
        return
    if left.shape == ():
        if not values_equal(left[()], right[()]):
            raise ValueError(f"Scalar data changed at {path}")
        return
    for start in range(0, left.shape[0], 128):
        if not values_equal(left[start : start + 128], right[start : start + 128]):
            raise ValueError(f"Data changed at {path}, row {start}")


def verify_conversion(source, converted):
    with h5py.File(source, "r") as left, h5py.File(converted, "r") as right:
        compare_unchanged(left, right)
        for path in INDEX_PATHS:
            before = read_nullable_index(left[path])
            node = right[path]
            if not isinstance(node, h5py.Dataset):
                raise ValueError(f"{path} was not converted to a dataset")
            if node.attrs.get("encoding-type") != "string-array":
                raise ValueError(f"{path} has an unexpected output encoding")
            if node.asstr()[:].tolist() != before:
                raise ValueError(f"{path} values or order changed")


def create_copy(inputs, plan_path, output):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 201 * 1024**3:
        raise OSError("Insufficient space above the 200 GiB reserve")
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_data = inputs / "data" / "replogle.h5ad"
    source_controls = inputs / "independent-reference-controls.h5ad"
    checks = {
        plan_path: manifest["plan_sha256"],
        source_data: manifest["dataset_sha256"],
        source_controls: manifest["reference_controls_sha256"],
        inputs / "split.toml": manifest["split_sha256"],
    }
    if manifest["status"] != "complete":
        raise ValueError("Frozen source inputs are not complete")
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen source hash mismatch: {path.name}")

    output.mkdir(mode=0o700)
    data_dir = output / "data"
    data_dir.mkdir()
    destination = data_dir / "replogle.h5ad"
    controls = output / "independent-reference-controls.h5ad"
    shutil.copyfile(source_data, destination)
    shutil.copyfile(source_controls, controls)
    with h5py.File(destination, "r+") as data:
        for path in INDEX_PATHS:
            convert_index(data, path)
        data.flush()
    verify_conversion(source_data, destination)
    if sha256(source_data) != manifest["dataset_sha256"]:
        raise ValueError("Frozen source data changed during conversion")
    if sha256(controls) != manifest["reference_controls_sha256"]:
        raise ValueError("Copied reference controls differ from frozen source")

    split = (
        f'[datasets]\nreplogle = "{data_dir}"\n\n[training]\nreplogle = "train"\n\n'
        '[fewshot."replogle.RPE1"]\n'
        f'val = {json.dumps(plan["targets"]["val"])}\n'
        f'test = {json.dumps(plan["targets"]["test"])}\n'
    )
    split_path = output / "split.toml"
    split_path.write_text(split, encoding="utf-8", newline="\n")
    result = dict(manifest)
    result.update(
        dataset_sha256=sha256(destination),
        split_sha256=sha256(split_path),
        source_manifest_sha256=sha256(manifest_path),
        source_dataset_sha256=manifest["dataset_sha256"],
        compatibility={
            "changed_hdf5_paths": list(INDEX_PATHS),
            "change": "nullable-string-array to string-array; values and order unchanged",
            "verification": "all other HDF5 dataset values, types, shapes and attributes compared",
        },
    )
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    create_copy(args.inputs, args.plan, args.output)


if __name__ == "__main__":
    main()
