# HPI2-NN: Surrogate Model for Pellet Fuelling in Tokamak Discharges

Author: Alex Panera Alvarez

## 🧩 Overview

**HPI2-NN** is a machine learning surrogate model of the **HPI2 pellet ablation and deposition code** (https://gitlab.com/hpi2_group/HPI2_code), developed to accelerate integrated modeling of pellet-fuelled tokamak discharges.

The model learns from around 10000 **HPI2 simulations** on WEST experimental data and ITER simulation data, and predicts **pellet deposition profiles** based on plasma parameters and pellet injection conditions.
It is designed for stand-alone and coupling into integrated modeling frameworks, enabling fast inference within integrated plasma scenario modeling.

---

## 🚀 Key Features

- Neural network surrogate trained on HPI2 synthetic data
- PCA compression of temperature and density profiles
- Supports ONNX inference for fast runtime execution
- Modular code for evaluation

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/DIFFER-NL/hpi2nn.git
cd hpi2nn
```

(Optional) Create a virtual environment and install dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install onnxruntime
```

The onnxruntime version used for this project is 1.22.0

### Install as a package

Install in editable mode to make the models importable from anywhere
(dependencies numpy, jax, onnx, onnxruntime, jaxonnxruntime, scipy are
installed automatically, see `pyproject.toml`):

```bash
pip install -e .
```

Note: only the **editable** install (`-e`) is supported. A regular install
would not ship the ONNX models and scalers in `artifacts_hpi2nn/`, which are
loaded with repository-relative paths.

## Quick Inference Example

Shown on inference/simple_inference.py
Advised to be run from hpi2nn/ folder

```bash
python inference_hpi2nn/simple_inference.py
```

Inputs: Pellet radius in m, velocity in m/s, Te and Ti profiles in eV, ne profile in m-3, B0 in T, first point (R1,Z1) and second point (R2,Z2) in m
x coord preferred in rho_tor_norm, but using a_norm will not impact too much the result.
The profiles can be given on any radial grid: they are interpolated onto the 101-point grid the models were trained on before any feature (PCA coefficients, Ti/Te fit) is computed
B0 is suppose to be negative always (inforced anyway in inference)

Outputs: deposition profile dne (m-3) and temperature change profile dTe (eV) same x coord as given in input, and the ablation time t_abl (s, never below 0.1 ms)

dTe is not a separate network output: it follows from dne by conserving the electron pressure, Te' = ne Te / (ne + dne). HPI2's own dTe follows the same relation to within 0.3-2.7% of its trough depth on the training databases, so dTe is as accurate as dne and always consistent with it.

For JETTO implementation--> inference_hpi2nn/HPI2-NN_JETTO.py
FOR JETTO multiply ne by 1e6

### IMAS inference

`src_hpi2nn/interfaces/imas_interface.py` runs HPI2-NN on a pellet stored in an IMAS data
entry and writes the result to another entry. It is the HPI2-NN counterpart of HPI2's
`hpi2_imas_interface.m` and takes the same arguments. Like HPI2's interface, it computes one pellet per call.
It needs IMAS-Python and, for the HDF5 and MDSplus backends, the IMAS access layer:

```bash
pip install -e ".[imas]"        # imas-python and imas-core
```

The quickest way in is `inference_hpi2nn/HPI2-NN_IMAS_inference.py`: fill in its fields (time,
input entry, output entry, pellet) and run it from hpi2nn/:

```bash
python inference_hpi2nn/HPI2-NN_IMAS_inference.py
```

The same from the command line, in hpi2nn/ (or `python -m hpi2nn.src_hpi2nn.interfaces.imas_interface`
from anywhere, with the package installed):

```bash
# entry in the IMAS public folder (~<user>/public/imasdb/west/3/54304/0), result in run 1
python -m src_hpi2nn.interfaces.imas_interface --time -37.5 --shot 54304 --run 0 --tokamak west --occ-pellets 1

# entry anywhere on disk (the folder holding master.h5), result in run 333, pellets occurrence 5
python -m src_hpi2nn.interfaces.imas_interface --time -37.5 --path <imasdb>/west/3/54304/0 --occ-pellets 1 \
    --out-run 333 --out-occ-pellets 5
```

- `--time`: time of the computation (s). A negative time takes the last pellet launched at or
  before `abs(time)`.
- Input entry: `--uri`, `--path`, or `--shot`, `--run`, `--tokamak`, `--user`, `--version`,
  `--backend`. With the last form the access layer looks in the user's public folder. The
  `--occ-equilibrium`, `--occ-core-profiles` and `--occ-pellets` options select the occurrences
  (default 0).
- Output entry: `--out-run`, `--out-shot`, `--out-user`, `--out-tokamak`, or the whole
  location with `--out-path` or `--out-uri`; occurrences with `--out-occ-pellets` and
  `--out-occ-core-profiles`. What is not given is taken from the input entry, with the run
  incremented by 1 and the same `pellets` and `core_profiles` occurrences.
- `--pellet-rank`: which pellet of the time slice (1-based, as in HPI2).
- `--inj-value`: force the injection line instead of detecting it from the pellet path.
- `--no-write`: compute, but write nothing.
- `--plot`: draw n_e and T_e before and after the pellet.

From Python, the same call returns the two output IDSs and the HPI2-NN inputs and outputs:

```python
from hpi2nn.src_hpi2nn.interfaces.imas_interface import hpi2nn_imas_interface

pellets_out, core_profiles_out, result = hpi2nn_imas_interface(
    -37.5,
    {"shot": 54304, "run": 0, "tokamak": "west", "occurrence": {"pellets": 1}},   # input entry
    {"run": 333, "occurrence": {"pellets": 5}},                                    # output entry
)
```

HPI2-NN takes fewer inputs than HPI2. It reads:

- `pellets` (`get_slice` at the time): `velocity_initial`, `shape` (the equivalent-sphere radius,
  computed from `shape.type.index` exactly as HPI2 does) and the `path_geometry` first and second
  points, from which the injection line is detected;
- `core_profiles`: `electrons.density_thermal` (else `electrons.density`),
  `electrons.temperature`, `t_i_average` and `q` (else the equilibrium q), on
  `grid.rho_tor_norm`;
- `equilibrium`: `vacuum_toroidal_field.b0`, and the plasma boundary to place the pellet path.

The 2D equilibrium, Z_eff and the code parameters that HPI2 uses are not inputs of HPI2-NN. The
plasma and pellet composition are only checked against the training databases (deuterium
pellets; D plasmas on WEST, D or DT on ITER), with a warning when they differ.

It writes, with `put_slice` as HPI2:

- `pellets`: the input pellets time slice, dated at the end of ablation. HPI2-NN predicts the
  ablation time, not the profiles along the path. So `path_profiles` holds only the two ends of
  the in-plasma path, where the pellet enters the plasma and where it is fully ablated, as
  `distance`, `position` and `velocity`.
- `core_profiles`: a new IDS on the input grid, with `electrons.density` and `density_thermal`
  = n_e + dne, `electrons.temperature` = T_e + dTe, the electron pressures, and
  `t_i_average` = T_i n_e / (n_e + dne). That is the dilution that gives dTe, applied to T_i as
  HPI2 applies the same mixing to both temperatures.
- no `temporary` IDS, where HPI2 stores its internal output.

HPI2's ablation time counts from the launch point, while HPI2-NN's counts only the time inside
the plasma (plus 0.1 ms). To date the output as HPI2 does, the interface adds the flight from
the launch point to the plasma boundary of the equilibrium. `code.parameters` records both
times, the injection line and the inputs used. The `ids_properties.comment` records any
out-of-domain warning. `put_slice` writes only time-dependent data, so in an entry holding
several pellets these two fields keep the values of the first pellet written, as with HPI2.

The interface refuses to write into the input entry, and refuses to add a slice at or before
the last one already in the output occurrence. Compute the pellets in time order, or write to
another run.

On Windows, `h5py` and `imas-core` cannot be imported in the same Python process (each ships
its own HDF5 library), so don't import h5py before calling the interface.

### Out-of-domain warnings

Each model ships its training ranges (`artifacts_hpi2nn/scalers/<device>/training_domain.json`):
per injection line, the minimum and maximum over its training cases of the pellet
velocity and volume, |B0|, ne and Te at rho = 0, 0.5 and 0.95, Ti/Te at rho = 0 and 0.95,
q at rho = 0.95 and the largest departure of Ti/Te from flat over rho <= 0.95 (the AUG
plasmas all have Ti = k Te, so a non-flat Ti/Te is flagged there), and of the network
inputs that describe the profile shapes (ne and Te PCA coefficients, Ti/Te fit,
rational-q surfaces). When an input falls outside,
`evaluate_model` warns once per call, listing each quantity, its value, the training
range and how far beyond the edge it is in units of the range. The outputs are not
changed. In held-out tests the error grew 1.2-1.5 times within a quarter of a range
beyond the edge, and several-fold further out; pellet sizes and velocities outside the
trained values gave errors of 13-34% of the peak.

The NumPy version issues an `HPI2NNOutOfDomainWarning` (a `UserWarning`):

```python
import warnings
from hpi2nn.src_hpi2nn.models.HPI2NN import HPI2NNOutOfDomainWarning
warnings.simplefilter("error", HPI2NNOutOfDomainWarning)    # stop instead of warning
warnings.simplefilter("ignore", HPI2NNOutOfDomainWarning)   # or silence it
```

The JAX version prints the same check with `jax.debug.print`, so that it also works under
`jit`. Values between the trained pellet velocities or sizes are inside the range and are
not flagged, although the model is less constrained there.

## 📈 Training and Data

HPI2-NN was trained using synthetic data from HPI2 simulations under various plasma conditions representative of WEST and ITER configurations.
A different NN has been trained for every injection line and tokamak.

Available lines: `WEST_upHFS`, `WEST_midHFS`, `WEST_LFS`, `ITER_upHFS`, `AUG_upHFS`.
The WEST lower-HFS (X-point) line is **withdrawn**: too few training cases, and the
wrong sign in its response to velocity, pellet size and temperature. That geometry is
still recognised and is refused with an explicit error rather than being served by a
neighbouring line.

The AUG model (`AUG_upHFS_v6.onnx`) takes 12 inputs instead of 14: B0 is left out,
because it barely varies in the AUG database, and Ti/Te enters as one number, its mean
over rho <= 0.95 instead of the two parameters of the exponential fit, because Ti/Te is
flat in every AUG plasma. B0 is still an argument of `evaluate_model` (ITER uses it, and
every line checks it against its training range) but does not change the AUG result. The
AUG model was trained only on plasmas with Ti proportional to Te, and warns when the
Ti/Te given departs from flat by more than its training plasmas do.

## ✅ Tests

`tests_hpi2nn/` holds the checks every released model must pass, before it is promoted to
`artifacts_hpi2nn/` and after:

1. the artifacts are complete, and the weights take the input vector `evaluate_model`
   builds (WEST 13, ITER 14, AUG 12 inputs);
2. accuracy on 50 reference HPI2 test cases per line: mean eps_prof and eps_t below a
   ceiling of 1.1 times the model the reference was exported from, so a new model may be
   at most 10% worse than the one it replaces;
3. frozen outputs on three cases, compared when the artifacts are the files they were
   exported from (a mismatch then means the inference code changed);
4. physical sanity: finite outputs, dne >= 0 with its peak inside the grid, dTe <= 0,
   t_abl >= 0.1 ms, and no reference case refused;
5. physics trends between the lowest and highest trained values, in at least 95% of the
   cases: a bigger pellet gives a higher peak and a longer ablation, a faster pellet a
   shorter ablation;
6. the injection line is detected from the geometry, and the withdrawn WEST lower-HFS
   line is refused (NumPy and JAX);
7. the JAX and NumPy versions agree to 1e-4 (where `jaxonnxruntime` is not installed,
   the JAX pipeline runs its network step on onnxruntime);
8. the out-of-domain warning stays silent on training cases and names the quantity
   pushed outside the range.

```bash
pip install -e ".[test]"
pytest                                   # the shipped artifacts
HPI2NN_ARTIFACTS=<bundle> pytest         # a candidate, before promoting it
```

A candidate bundle is a folder with `models/` and `scalers/<device>/`, laid out like
`artifacts_hpi2nn/`. The reference cases (`tests_hpi2nn/reference/<line>.npz`) are
exported from the HPI2 databases by `eval/accuracy/export_reference_cases.py` in the
training repository (hpi2nn-train-eval). Re-export them after promoting a model: that
updates the frozen outputs, and lowers the ceilings if the new model is better.

## 📘 Citation

A manuscript explaining and using this model is under preparation. So this repository is for the moment the only citable source.

## 👤 Author

Alex Panera Alvarez
PhD Candidate, Integrated Modelling Group — DIFFER
Email: a.paneraalvarez@differ.nl

GitHub: @alexpanera

## 📜 License

This project is licensed under the MIT License

.
© 2025 DIFFER — Dutch Institute for Fundamental Energy Research.

## 💡 Acknowledgements

This work was supported by EUROfusion under the Theory, Simulation, Verification and Validation (TSVV) tasks,
and carried out in collaboration with WEST and ITER Organization.

Special thanks to Florian Köchl, Eleonore Geulin for their contribution and the integrated modeling community for valuable discussions.
