"""tests for base normalization helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DATASET_ROOT = Path(__file__).parent.parent.parent
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from multigame.base import enforce_top_left_16x16


def test_enforce_keeps_16x16_without_warning():
    arr = np.ones((16, 16), dtype=np.int32)
    out = enforce_top_left_16x16(arr, game="dummy", source_id="x")
    assert out.shape == (16, 16)
    np.testing.assert_array_equal(out, arr)


def test_enforce_slices_and_warns_for_larger_shape():
    arr = np.arange(20 * 18, dtype=np.int32).reshape(20, 18)
    with pytest.warns(RuntimeWarning):
        out = enforce_top_left_16x16(arr, game="dummy", source_id="y")
    assert out.shape == (16, 16)
    np.testing.assert_array_equal(out, arr[:16, :16])


def test_enforce_pads_and_warns_for_smaller_shape():
    arr = np.full((10, 12), 7, dtype=np.int32)
    with pytest.warns(RuntimeWarning):
        out = enforce_top_left_16x16(arr, game="dummy", source_id="z")
    assert out.shape == (16, 16)
    np.testing.assert_array_equal(out[:10, :12], arr)
    assert np.all(out[10:, :] == 0)
    assert np.all(out[:, 12:] == 0)
