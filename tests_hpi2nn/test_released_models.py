# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

"""The checks every released HPI2-NN model must pass, before and after promotion.

1. Artifacts: weights, PCA, normalization and training domain present, and the weights
   take the input vector evaluate_model builds for the line (WEST 13, ITER 14, AUG 12).
2. Accuracy on 50 reference HPI2 test cases: mean eps_prof and eps_t below the line's
   ceilings (1.1 times the model the reference was exported from: a new model may be at
   most 10% worse than the one it replaces).
3. Frozen outputs: when the artifacts are the files the reference was exported from,
   three cases reproduce the stored outputs, so a change means the inference code changed.
4. Physical sanity of every output, and no reference case refused.
5. Physics trends between the lowest and highest trained values: a bigger pellet gives a
   higher peak and a longer ablation, a faster pellet a shorter ablation.
6. Injection-line selection from the geometry, and the withdrawn WEST lower-HFS line
   refused.
8. Out-of-domain guard: silent inside the training domain, warning outside it (and, on
   AUG, when T_i/T_e is not flat).

Test 7 (NumPy and JAX agree) is in test_jax.py.

Run from the hpi2nn folder::

    pytest tests_hpi2nn                                   # the shipped artifacts
    HPI2NN_ARTIFACTS=<bundle> pytest tests_hpi2nn         # a candidate, before promoting
"""

import hashlib
import json
import warnings

import numpy as np
import onnxruntime as ort
import pytest

from conftest import (HPI2NN, NORMALIZATION, RELEASED, WITHDRAWN_GEOMETRY, case_arguments,
                      numpy_outputs, radius_of, reference, run_numpy, candidate_under_test)

#: Test 5: share of the reference cases that must show each trend, and that must be
#: evaluable at both ends (not refused). HPI2's own trend is weak in some plasmas
#: (one ITER case: t_abl 0.217, 0.217, 0.233 ms over the three pellet sizes).
TREND_PASS = 0.95
TREND_EVALUABLE = 0.90
#: Test 3: tolerance of the frozen outputs, relative to their peak.
FROZEN_RTOL = 1e-5

LINES = pytest.mark.parametrize("released", list(RELEASED), indirect=True)


def artifact_files(root, line):
    device, weights, _ = RELEASED[line]
    scalers = root / "scalers" / device
    return [root / "models" / weights, scalers / NORMALIZATION[device],
            scalers / "pca_ne_data.npz", scalers / "pca_Te_data.npz"]


# -- 1. artifacts ---------------------------------------------------------------------

@LINES
def test_artifacts_complete_and_consistent(released, artifacts_under_test):
    line = released
    device, _, n_inputs = RELEASED[line]
    files = artifact_files(artifacts_under_test, line)
    domain_path = artifacts_under_test / "scalers" / device / "training_domain.json"
    missing = [str(p) for p in [*files, domain_path] if not p.is_file()]
    assert not missing, f"{line}: missing {missing}"

    session = ort.InferenceSession(str(files[0]))
    assert session.get_inputs()[0].shape[-1] == n_inputs, \
        f"{line}: the weights take {session.get_inputs()[0].shape[-1]} inputs, evaluate_model builds {n_inputs}"
    assert session.get_outputs()[0].shape[-1] == 7

    with np.load(files[1]) as norm:
        assert norm["scaler_X_mean"].shape == norm["scaler_X_std"].shape == (14,)
        assert norm["scaler_y_mean"].shape == norm["scaler_y_std"].shape == (7,)
    for pca_file in files[2:]:
        with np.load(pca_file) as pca:
            assert pca["components"].shape == (3, 101)

    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    assert line in domain["lines"], f"{line}: not in {domain_path.name}"
    assert len(domain["lines"][line]["inputs"]) == n_inputs


# -- 2. accuracy ----------------------------------------------------------------------

@LINES
def test_accuracy_on_reference_cases(released):
    line = released
    ref = reference(line)
    results = numpy_outputs(line)
    ok = [k for k, r in enumerate(results) if not isinstance(r, Exception)]
    dne = np.array([results[k][0] for k in ok])
    t_abl = np.array([float(results[k][2]) for k in ok])
    truth, t_truth = ref["dne_hpi2"][ok], ref["t_abl_hpi2"][ok]
    eps_prof = float(np.mean(np.sqrt(np.mean((dne - truth) ** 2, axis=1)) / truth.max(axis=1)))
    eps_t = float(np.mean(np.abs(t_abl - t_truth) / t_truth))
    assert eps_prof <= ref["ceiling_eps_prof"], \
        f"{line}: eps_prof {100 * eps_prof:.2f}% above the ceiling {100 * ref['ceiling_eps_prof']:.2f}%"
    assert eps_t <= ref["ceiling_eps_t"], \
        f"{line}: eps_t {100 * eps_t:.2f}% above the ceiling {100 * ref['ceiling_eps_t']:.2f}%"


# -- 3. frozen outputs ------------------------------------------------------------------

@LINES
def test_frozen_outputs(released, artifacts_under_test):
    line = released
    ref = reference(line)
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in artifact_files(artifacts_under_test, line)}
    if digests != ref["meta"]["artifact_sha256"]:
        message = (f"{line}: the artifacts under test are not the files the frozen outputs "
                   "were exported from")
        if candidate_under_test():
            pytest.skip(message + " (expected for a candidate)")
        pytest.fail(message + "; re-export the reference cases after promoting "
                    "(eval/accuracy/export_reference_cases.py)")
    for k in range(len(ref["frozen_t_abl"])):
        dne, dte, t_abl = numpy_outputs(line)[k]
        np.testing.assert_allclose(dne, ref["frozen_dne"][k], rtol=FROZEN_RTOL,
                                   atol=FROZEN_RTOL * np.abs(ref["frozen_dne"][k]).max())
        np.testing.assert_allclose(dte, ref["frozen_dte"][k], rtol=FROZEN_RTOL,
                                   atol=FROZEN_RTOL * np.abs(ref["frozen_dte"][k]).max())
        np.testing.assert_allclose(t_abl, ref["frozen_t_abl"][k], rtol=FROZEN_RTOL)


# -- 4. physical sanity -----------------------------------------------------------------

@LINES
def test_outputs_physical(released):
    line = released
    ref = reference(line)
    for k, result in enumerate(numpy_outputs(line)):
        assert not isinstance(result, Exception), f"{line} case {k} refused: {result}"
        dne, dte, t_abl = result
        assert np.isfinite(dne).all() and np.isfinite(dte).all() and np.isfinite(t_abl), \
            f"{line} case {k}: non-finite output"
        assert dne.min() >= 0.0, f"{line} case {k}: negative dne"
        peak = int(np.argmax(dne))
        assert 0 < peak < len(dne) - 1, f"{line} case {k}: peak on the grid edge (rho = {ref['rho'][peak]})"
        assert dte.max() <= 1e-9 * np.abs(ref["Te"][k]).max(), f"{line} case {k}: the pellet heats"
        assert t_abl >= HPI2NN.T_ABL_MIN, f"{line} case {k}: t_abl below 0.1 ms"


# -- 5. physics trends ------------------------------------------------------------------

@LINES
def test_physics_trends(released):
    line = released
    ref = reference(line)
    radii = [radius_of(v) for v in (ref["trained_volumes"].min(), ref["trained_volumes"].max())]
    speeds = [float(v) for v in (ref["trained_velocities"].min(), ref["trained_velocities"].max())]
    n = len(ref["t_abl_hpi2"])
    passed = {"peak rises with size": 0, "t_abl rises with size": 0, "t_abl falls with velocity": 0}
    evaluable = {"size": 0, "velocity": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(n):
            try:
                small, big = (run_numpy(**case_arguments(ref, k, line, pellet_radius=r)) for r in radii)
                evaluable["size"] += 1
                passed["peak rises with size"] += int(big[0].max() > small[0].max())
                passed["t_abl rises with size"] += int(big[2] > small[2])
            except ValueError:                 # refused: more than half the profile negative
                pass
            try:
                slow, fast = (run_numpy(**case_arguments(ref, k, line, vel_value=v)) for v in speeds)
                evaluable["velocity"] += 1
                passed["t_abl falls with velocity"] += int(fast[2] < slow[2])
            except ValueError:
                pass
    for axis, count in evaluable.items():
        assert count >= TREND_EVALUABLE * n, f"{line}: refused at the trained {axis} extremes in {n - count} of {n} cases"
    for trend, count in passed.items():
        total = evaluable["size" if "size" in trend else "velocity"]
        assert count >= TREND_PASS * total, f"{line}: '{trend}' holds in only {count} of {total} cases"


# -- 6. injection-line selection --------------------------------------------------------

@LINES
def test_line_detected_from_geometry(released):
    line = released
    ref = reference(line)
    for k in range(len(ref["t_abl_hpi2"])):
        detected = HPI2NN.find_closest_injection_line(list(ref["first_point"][k]),
                                                      list(ref["second_point"][k]))
        assert detected == line, f"case {k}: geometry detected as {detected}, not {line}"


def test_withdrawn_line_refused():
    ref = reference("WEST_upHFS")
    by_geometry = case_arguments(ref, 0, None, first_point=WITHDRAWN_GEOMETRY[0],
                                 second_point=WITHDRAWN_GEOMETRY[1])
    with pytest.raises(ValueError, match="withdrawn"):
        run_numpy(**by_geometry)
    with pytest.raises(ValueError, match="withdrawn"):
        run_numpy(**case_arguments(ref, 0, "WEST_lowHFS"))


# -- 8. out-of-domain guard -------------------------------------------------------------

def ood_warnings(**arguments) -> list[str]:
    """The out-of-domain warnings of one call (the call may still refuse the input)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run_numpy(**arguments)
        except ValueError:
            pass
    return [str(w.message) for w in caught if issubclass(w.category, HPI2NN.HPI2NNOutOfDomainWarning)]


@LINES
def test_guard_silent_inside_domain(released):
    line = released
    ref = reference(line)
    for k in range(len(ref["in_domain_radius"])):
        found = ood_warnings(**case_arguments(ref, k, line, prefix="in_domain_"))
        assert not found, f"{line} training case {k} flagged:\n{found[0]}"


#: Inputs pushed to 1.5 times the top of their training range, and the quantity the
#: warning must name. Each entry: (domain key, how to apply a factor to the input).
PUSHED_OUT = {
    "pellet velocity": ("v_pellet", lambda case, top: {"vel_value": top}),
    "pellet volume": ("V_pellet", lambda case, top: {"pellet_radius": radius_of(top)}),
    "n_e at rho = 0": ("ne_0", lambda case, top: {"ne": case["ne"] * top / case["ne"][0]}),
    "T_e at rho = 0": ("Te_0", lambda case, top: {"Te": case["Te"] * top / case["Te"][0]}),
    "T_i/T_e at rho = 0": ("TiTe_0", lambda case, top: {
        "Ti": case["Ti"] * top / (case["Ti"][0] / case["Te"][0])}),
    "q at rho = 0.95": ("q95", lambda case, top: {
        "q": case["q"] * top / np.interp(0.95, case["rho"], case["q"])}),
}


@LINES
@pytest.mark.parametrize("quantity", list(PUSHED_OUT))
def test_guard_warns_outside_domain(released, quantity, artifacts_under_test):
    line = released
    ref = reference(line)
    device = RELEASED[line][0]
    domain = json.loads((artifacts_under_test / "scalers" / device / "training_domain.json")
                        .read_text(encoding="utf-8"))
    key, push = PUSHED_OUT[quantity]
    top = 1.5 * domain["lines"][line]["ranges"][key][1]
    case = {name: ref["in_domain_" + name][0] for name in ("ne", "Te", "Ti", "q")}
    case["rho"] = ref["rho"]
    arguments = case_arguments(ref, 0, line, prefix="in_domain_", **push(case, top))
    found = ood_warnings(**arguments)
    assert len(found) == 1, f"{line}: {len(found)} out-of-domain warnings for {quantity}"
    assert quantity in found[0], f"{line}: the warning does not name {quantity}:\n{found[0]}"


def test_aug_warns_when_ti_te_not_flat():
    """AUG's plasmas all have T_i = k T_e: a T_i/T_e rising by +/-20% across the profile is
    outside its training domain and must be named in the warning."""
    ref = reference("AUG_upHFS")
    ti = ref["in_domain_Ti"][0] * (0.8 + 0.4 * ref["rho"])
    found = ood_warnings(**case_arguments(ref, 0, "AUG_upHFS", prefix="in_domain_", Ti=ti))
    assert len(found) == 1 and "departure from flat" in found[0], found
