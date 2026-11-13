"""Tests for the frozen input index-only compatibility conversion."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np


path = Path(__file__).with_name("convert-state-pilot-cellload-index-v1.py")
spec = importlib.util.spec_from_file_location("convert_pilot", path)
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


def write_fixture(path, masked=False):
    with h5py.File(path, "w") as data:
        data.create_dataset("X", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype="f4"))
        data.create_dataset("null", dtype="f4")
        for parent, values in (("obs", ["a", "b"]), ("var", ["A", "B"])):
            group = data.create_group(parent).create_group("_index")
            group.attrs["encoding-type"] = "nullable-string-array"
            group.create_dataset("values", data=np.asarray(values, dtype=object), dtype=h5py.string_dtype())
            group.create_dataset("mask", data=np.array([False, masked]))


class ConversionTests(unittest.TestCase):
    def test_only_index_encoding_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.h5"
            result = Path(tmp) / "result.h5"
            write_fixture(source)
            shutil.copyfile(source, result)
            with h5py.File(result, "r+") as data:
                for path in convert.INDEX_PATHS:
                    convert.convert_index(data, path)
            convert.verify_conversion(source, result)
            with h5py.File(result, "r+") as data:
                data["X"][0, 0] = 9
            with self.assertRaisesRegex(ValueError, "Data changed"):
                convert.verify_conversion(source, result)

    def test_null_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.h5"
            write_fixture(source, masked=True)
            with h5py.File(source, "r+") as data:
                with self.assertRaisesRegex(ValueError, "missing"):
                    convert.convert_index(data, "obs/_index")


if __name__ == "__main__":
    unittest.main()
