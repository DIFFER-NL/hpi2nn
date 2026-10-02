# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

"""Shared fixtures of the HPI2-NN test suite.

Every released model is checked against reference HPI2 cases exported from the training
databases (``reference/<line>.npz``, written by
``eval/accuracy/export_reference_cases.py`` in hpi2nn-train-eval). By default the suite
tests the shipped artifacts (``artifacts_hpi2nn/``). To test a candidate before promoting
it, point ``HPI2NN_ARTIFACTS`` at its bundle, a folder holding ``models/`` and
``scalers/<device>/``; the lines of devices it has no weights for are then skipped.
"""

import contextlib
import functools
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"

try:
    from hpi2nn.src_hpi2nn.models import HPI2NN
except ImportError:                    # not pip-installed: import from the checkout
    sys.path.insert(0, str(HERE.parent.parent))
    from hpi2nn.src_hpi2nn.models import HPI2NN

#: The released models: injection line -> (device, weights file, number of network inputs).
RELEASED = {
    "WEST_upHFS": ("WEST", "WEST_upHFS_noBo_v4.onnx", 13),
    "WEST_midHFS": ("WEST", "WEST_midHFS_noBo_v4.onnx", 13),
    "WEST_LFS": ("WEST", "WEST_LFS_noBo_v4.onnx", 13),
    "ITER_upHFS": ("ITER", "ITER_upHFS_v4.onnx", 14),
    "AUG_upHFS": ("AUG", "AUG_upHFS_v5.onnx", 12),
}
#: Normalization file per device, as evaluate_model loads it.
NORMALIZATION = {"WEST": "Normalization_v4.npz", "ITER": "Normalization_v4.npz",
                 "AUG": "Normalization_AUG_A2_v3.npz"}
#: Geometry of the withdrawn WEST lower-HFS (X-point) line.
WITHDRAWN_GEOMETRY = ([1.8, -0.33], [2.7336, -0.6884])

ARTIFACTS_ENV = "HPI2NN_ARTIFACTS"


def candidate_under_test() -> bool:
    """True when the suite runs on a candidate bundle rather than the shipped artifacts."""
    return bool(os.environ.get(ARTIFACTS_ENV))


@pytest.fixture(scope="session", autouse=True)
def artifacts_under_test():
    """The artifact tree under test; HPI2NN.py is pointed at it for the whole session."""
    if not candidate_under_test():
        yield HPI2NN.WEIGHTS_PATH.parent
        return
    root = Path(os.environ[ARTIFACTS_ENV]).resolve()
    previous = (HPI2NN.WEIGHTS_PATH, HPI2NN.SCALERS_PATH)
    HPI2NN.WEIGHTS_PATH, HPI2NN.SCALERS_PATH = root / "models", root / "scalers"
    try:
        yield root
    finally:
        HPI2NN.WEIGHTS_PATH, HPI2NN.SCALERS_PATH = previous


@pytest.fixture
def released(request, artifacts_under_test):
    """The line of a test parametrized over RELEASED, skipped if a candidate bundle under
    test has no weights for it."""
    line = request.param
    _, weights, _ = RELEASED[line]
    if candidate_under_test() and not (artifacts_under_test / "models" / weights).is_file():
        pytest.skip(f"{line}: not among the candidate artifacts in {artifacts_under_test}")
    return line


@functools.lru_cache(maxsize=None)
def reference(line: str) -> dict:
    """The exported reference cases of a line (see the module docstring)."""
    with np.load(REFERENCE / f"{line}.npz") as data:
        ref = {name: data[name] for name in data.files}
    ref["meta"] = json.loads(str(ref["meta"]))
    return ref


def case_arguments(ref: dict, k: int, line: str, prefix: str = "", **override) -> dict:
    """evaluate_model's arguments for reference case k (``prefix='in_domain_'`` for the
    training cases inside the domain), with any of them overridden."""
    arguments = {
        "pellet_radius": float(ref[prefix + "radius"][k]),
        "vel_value": float(ref[prefix + "velocity"][k]),
        "x_coord": ref["rho"], "Te": ref[prefix + "Te"][k], "ne": ref[prefix + "ne"][k],
        "Ti": ref[prefix + "Ti"][k], "q": ref[prefix + "q"][k], "B0": float(ref[prefix + "B0"][k]),
        "inj_value": line,
    }
    arguments.update(override)
    return arguments


def run_numpy(**arguments):
    """HPI2NN.evaluate_model with its progress prints swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return HPI2NN.evaluate_model(**arguments)


@functools.lru_cache(maxsize=None)
def numpy_outputs(line: str) -> tuple:
    """(dne, dTe, t_abl), or the exception raised, for every reference case of a line."""
    ref = reference(line)
    results = []
    for k in range(len(ref["t_abl_hpi2"])):
        try:
            results.append(run_numpy(**case_arguments(ref, k, line)))
        except Exception as exc:             # noqa: BLE001 - reported by the sanity test
            results.append(exc)
    return tuple(results)


def radius_of(volume: float) -> float:
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
