import numpy as np
import pytest

from epi_c3_foundation_stage1_sources_v1 import windows


@pytest.mark.parametrize("length", [2, 47, 48, 49, 50, 127, 3072, 3073, 5000])
def test_source_windows_are_deterministic_and_cover_every_row_once(length):
    actual = windows(length, 48, 91)
    assert actual == windows(length, 48, 91)
    assert all(2 <= right - left <= 49 for left, right in actual)
    assert sorted(index for left, right in actual for index in range(left, right)) == list(range(length))


def test_windows_reject_invalid_source():
    with pytest.raises(ValueError):
        windows(1, 48, 1)
