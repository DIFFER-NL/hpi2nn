# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

"""Test 7: the JAX version (JAX_HPI2NN.py) agrees with the NumPy one; and the JAX side
of tests 6 (the withdrawn line is refused) and 8 (the out-of-domain guard).

jaxonnxruntime is the JAX version's network backend. Where it is not installed, the
network step runs on onnxruntime instead, so that everything else in the JAX pipeline
(PCA projection, T_i/T_e fit, rational-q surfaces, normalization, reconstruction,
guards) is still compared with the NumPy version.
"""

import contextlib
import importlib.util
import io

import numpy as np
import onnxruntime as ort
import pytest

jnp = pytest.importorskip("jax.numpy")

from conftest import RELEASED, case_arguments, numpy_outputs, reference   # noqa: E402
from hpi2nn.src_hpi2nn.models import JAX_HPI2NN                            # noqa: E402

#: Reference cases compared per line, and the agreement required, relative to the peak
#: of the NumPy output (to t_abl itself for t_abl).
N_CASES = 5
TOLERANCE = 1e-4

LINES = pytest.mark.parametrize("released", list(RELEASED), indirect=True)


def onnxruntime_infer_fn(path):
    session = ort.InferenceSession(path)
    name = session.get_inputs()[0].name
    return lambda x: session.run(None, {name: np.asarray(x, dtype=np.float32)})[0]


@pytest.fixture(scope="module", autouse=True)
def jax_artifacts(artifacts_under_test):
    """Point JAX_HPI2NN.py at the artifacts under test (and at onnxruntime if needed)."""
    previous = (JAX_HPI2NN.WEIGHTS_PATH, JAX_HPI2NN.SCALERS_PATH, JAX_HPI2NN.get_onnx_infer_fn)
    JAX_HPI2NN.WEIGHTS_PATH = artifacts_under_test / "models"
    JAX_HPI2NN.SCALERS_PATH = artifacts_under_test / "scalers"
    if importlib.util.find_spec("jaxonnxruntime") is None:
        JAX_HPI2NN.get_onnx_infer_fn = onnxruntime_infer_fn
    try:
        yield
    finally:
        JAX_HPI2NN.WEIGHTS_PATH, JAX_HPI2NN.SCALERS_PATH, JAX_HPI2NN.get_onnx_infer_fn = previous


def run_jax(**arguments):
    """JAX evaluate_model on NumPy inputs: (dne, dTe, t_abl, everything it printed)."""
    arguments = {name: jnp.asarray(value) if isinstance(value, np.ndarray) else value
                 for name, value in arguments.items()}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        dne, dte, t_abl = JAX_HPI2NN.evaluate_model(**arguments)
    return np.asarray(dne), np.asarray(dte), float(t_abl), printed.getvalue()


@LINES
def test_numpy_and_jax_agree(released):
    line = released
    ref = reference(line)
    for k in range(N_CASES):
        dne_np, dte_np, t_np = numpy_outputs(line)[k]
        dne_jx, dte_jx, t_jx, _ = run_jax(**case_arguments(ref, k, line))
        assert np.abs(dne_jx - dne_np).max() <= TOLERANCE * dne_np.max(), \
            f"{line} case {k}: dne differs by {np.abs(dne_jx - dne_np).max() / dne_np.max():.2e} of the peak"
        assert np.abs(dte_jx - dte_np).max() <= TOLERANCE * np.abs(dte_np).max(), \
            f"{line} case {k}: dTe differs by {np.abs(dte_jx - dte_np).max() / np.abs(dte_np).max():.2e} of the trough"
        assert abs(t_jx - t_np) <= TOLERANCE * t_np, \
            f"{line} case {k}: t_abl differs by {abs(t_jx - t_np) / t_np:.2e}"


def test_jax_refuses_withdrawn_line():
    with pytest.raises(ValueError, match="withdrawn"):
        run_jax(**case_arguments(reference("WEST_upHFS"), 0, "WEST_lowHFS"))


@LINES
def test_jax_guard(released):
    line = released
    ref = reference(line)
    *_, printed = run_jax(**case_arguments(ref, 0, line, prefix="in_domain_"))
    assert "outside the training range" not in printed, f"{line}: training case flagged:\n{printed}"
    faster = 1.5 * float(ref["trained_velocities"].max())
    *_, printed = run_jax(**case_arguments(ref, 0, line, prefix="in_domain_", vel_value=faster))
    assert "outside the training range" in printed and "pellet velocity" in printed, \
        f"{line}: no warning for a pellet at {faster:.0f} m/s:\n{printed}"


def test_jax_aug_warns_when_ti_te_not_flat():
    ref = reference("AUG_upHFS")
    ti = ref["in_domain_Ti"][0] * (0.8 + 0.4 * ref["rho"])
    *_, printed = run_jax(**case_arguments(ref, 0, "AUG_upHFS", prefix="in_domain_", Ti=ti))
    assert "departure from flat" in printed, printed
