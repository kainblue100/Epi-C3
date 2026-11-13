import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("public_stage1_driver", ROOT / "train-rna-foundation-stage1-v1.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_driver_uses_explicit_data_root_and_bundled_helpers(tmp_path, monkeypatch):
    loaded = []

    def fake_load(path, name):
        loaded.append((path, name))
        return object()

    monkeypatch.setattr(driver, "load_script", fake_load)
    with pytest.raises(FileNotFoundError) as error:
        driver.build_sources(tmp_path)
    assert Path(error.value.filename) == (
        tmp_path / "c3-vcell" / "runs" / "continuation-20261111-v1" / "arc-stream-full-v1" / "plan.json"
    )
    assert [path.parent for path, _ in loaded] == [ROOT, ROOT]
    assert [path.name for path, _ in loaded] == [
        "stream-arc-training-v1.py", "stream-orion-representation-v1.py"
    ]
