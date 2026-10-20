# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.
"""HPI2-NN for IMAS: read a pellet and its plasma from an IMAS data entry, run HPI2-NN and
write the result to an IMAS data entry.

This is the HPI2-NN counterpart of HPI2's ``hpi2_imas_interface.m``, and follows it step by
step: same input references (shot, run, user, tokamak, version, backend and occurrences),
same time handling, same IDSs read and written. Like the HPI2 interface, it computes one
pellet per call.

Usage
-----
From Python, with the package installed (``pip install -e ".[imas]"``)::

    from hpi2nn.src_hpi2nn.interfaces.imas_interface import hpi2nn_imas_interface

    pellets_out, core_profiles_out, result = hpi2nn_imas_interface(
        -37.5, {"shot": 54304, "run": 0, "tokamak": "west", "occurrence": {"pellets": 1}})

``inference_hpi2nn/HPI2-NN_IMAS_inference.py`` makes the same call with every field written out.

From the command line, in hpi2nn/ (or ``python -m hpi2nn.src_hpi2nn.interfaces.imas_interface``
from anywhere, with the package installed)::

    # entry in the IMAS public folder (~<user>/public/imasdb/west/3/54304/0), result in run 1
    python -m src_hpi2nn.interfaces.imas_interface --time -37.5 --shot 54304 --run 0 --tokamak west --occ-pellets 1

    # entry anywhere on disk (the folder holding master.h5), result in the sibling run folder
    python -m src_hpi2nn.interfaces.imas_interface --time -37.5 --path <imasdb>/west/3/54304/0 --occ-pellets 1

    # any IMAS URI
    python -m src_hpi2nn.interfaces.imas_interface --time -37.5 --uri "imas:hdf5?path=..." --out-uri "..."

Inputs
------
``time``: time of the computation (s). If negative, the computation is synchronised with
the pellets IDS, as in HPI2: it uses the last pellet launched at or before ``abs(time)``
according to ``time_slice[k].time`` (``pellets.time`` if the pellets IDS has a homogeneous
time base), and that pellet's time becomes the time of the computation. If positive, the
pellet is the time slice closest to ``time``. The pellet's slice time is its launch time.

``imas_ref_in``: the input entry, given by ``uri``, by ``path`` (the folder of the run), or
by ``shot``, ``run``, ``user``, ``tokamak``, ``version`` and ``backend``, in which case the
access layer looks for it in the user's public folder (``~<user>/public/imasdb``), as the
HPI2 interface does. ``occurrence`` selects the ``equilibrium``, ``core_profiles`` and
``pellets`` occurrences (default 0).

``imas_ref_out``: the output entry. Missing fields are taken from the input with the run
incremented by 1; the occurrences of ``pellets`` and ``core_profiles`` default to the input
ones. Writing into the input entry itself is refused.

``pellet_rank``: which pellet of the time slice to compute (1-based, as in HPI2).

Read (``get_slice`` at ``time``)
--------------------------------
``pellets`` and ``equilibrium`` are read at the closest sample, as HPI2 does. ``core_profiles``
is read at the last sample at or before ``time`` (HPI2: the closest), so that the input
plasma is never one already perturbed by the pellet; the time of that sample is recorded in
the output (``code.parameters``, ``ids_properties.comment``).

HPI2-NN takes fewer inputs than HPI2. From the IDSs it uses:

* ``pellets``: ``velocity_initial``; ``shape`` (the equivalent-sphere radius, computed from
  ``shape.type.index`` and ``shape.size`` exactly as HPI2 does); ``path_geometry`` first and
  second points, from which the injection line is detected.
* ``core_profiles``: on ``grid.rho_tor_norm``, ``electrons.density_thermal`` (else
  ``electrons.density``), ``electrons.temperature``, ``t_i_average`` and ``q`` (else the
  ``equilibrium`` q).
* ``equilibrium``: ``vacuum_toroidal_field.b0`` (else the ``core_profiles`` one), and the
  plasma boundary (``boundary.outline``), only to place the pellet path (see *Time*).

HPI2 also uses the 2D equilibrium, Z_eff, the plasma and pellet composition and its code
parameters; HPI2-NN does not take them. The composition is only checked against the
training databases (deuterium pellets; D plasmas on WEST, D or DT on ITER), with a warning
if it differs.

Written (``put_slice``, as HPI2)
--------------------------------
* ``pellets``: the input pellets time slice. HPI2 fills ``path_profiles`` along the whole
  in-plasma path. HPI2-NN predicts the ablation time, not the profiles along the path, so only
  the two ends of the in-plasma path are written to ``path_profiles``, where the pellet enters
  the plasma and where it is fully ablated. That means ``distance``, ``position`` and
  ``velocity``. ``n_e``, ``t_e``, ``rho_tor_norm``, ``psi``, ``ablation_rate``,
  ``ablated_particles`` and ``rho_tor_norm_drift`` stay empty.
* ``core_profiles``: a new IDS on the input ``core_profiles`` grid. ``electrons.density`` and
  ``density_thermal`` are n_e + Δn_e, ``electrons.temperature`` is T_e + ΔT_e, and
  ``pressure`` and ``pressure_thermal`` follow from those. ``t_i_average`` is
  T_i n_e/(n_e + Δn_e). That is the dilution that gives ΔT_e in HPI2-NN. HPI2 updates T_i
  with the same mixing as T_e, so HPI2-NN applies it to T_i too.
* No ``temporary`` IDS: HPI2 stores its internal output there, and HPI2-NN has none.

Time
----
HPI2's ablation time counts from the launch point (``path_geometry.first_point``), so it
includes the flight to the plasma edge. HPI2-NN's ablation time counts only the time inside
the plasma (plus 0.1 ms), because the flight to the edge depends on where the equilibrium
places the plasma, which the network does not see. To date the output as HPI2 does (pellet
time + ablation time from launch), this interface adds the flight time to the edge,
computed from the equilibrium boundary along the line from the first to the second point.
The ``path_profiles.distance`` it writes is measured from the launch point, as HPI2's.

Requirements
------------
IMAS-Python and, for the HDF5 and MDSplus backends, the IMAS access layer::

    pip install -e ".[imas]"        # from hpi2nn/; or: pip install imas-python imas-core
"""
import argparse
import contextlib
import copy
import datetime
import getpass
import io
import subprocess
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import imas
except ImportError as err:
    raise ImportError(
        "the HPI2-NN IMAS interface needs IMAS-Python and the IMAS access layer: "
        'pip install -e ".[imas]" in hpi2nn/, or pip install imas-python imas-core'
    ) from err

from ..models.HPI2NN import evaluate_model, find_closest_injection_line

# the hpi2nn repository (src_hpi2nn/interfaces/ -> two levels up), for its git commit and version
project_root = Path(__file__).resolve().parents[2]

E_CHARGE = 1.602176634e-19   # C, converts eV to J for the pressure
T_HOMOG = 1e-4               # s, the 0.1 ms that the ablation time adds to the flight in the plasma

# defaults of an IMAS reference, as in hpi2_imas_interface.m (user: the current user)
REF_DEFAULTS = {"run": 0, "user": None, "tokamak": "test", "version": "3", "backend": "hdf5"}

# compositions of the training databases (AUG's is not recorded)
TRAINED_PELLET = {"WEST": {"D"}, "ITER": {"D"}}
TRAINED_PLASMA = {"WEST": {"D"}, "ITER": {"D", "DT"}}

# core_profiles grid quantities copied from the input to the output
GRID_FIELDS = ("rho_tor_norm", "rho_tor", "rho_pol_norm", "psi", "volume", "area", "surface",
               "psi_magnetic_axis", "psi_boundary")


# ---------------------------------------------------------------------------------------
# IMAS references
# ---------------------------------------------------------------------------------------
def complete_ref_in(ref):
    """The input reference with its defaults and occurrences filled in."""
    ref = dict(ref or {})
    for key, value in REF_DEFAULTS.items():
        if ref.get(key) is None:
            ref[key] = value
    if ref["user"] is None:
        ref["user"] = getpass.getuser()
    occurrence = dict(ref.get("occurrence") or {})
    for name in ("equilibrium", "core_profiles", "pellets"):
        occurrence[name] = int(occurrence.get(name) or 0)
    ref["occurrence"] = occurrence
    if not ref.get("uri") and not ref.get("path") and ref.get("shot") is None:
        raise ValueError("shot number must be defined (or give the entry as uri or path)")
    return ref


def complete_ref_out(ref_in, ref_out):
    """The output reference: missing fields from the input, run incremented by 1."""
    out = dict(ref_out or {})
    occurrence = dict(out.get("occurrence") or {})
    for name in ("pellets", "core_profiles"):
        if occurrence.get(name) is None:
            occurrence[name] = ref_in["occurrence"][name]
        occurrence[name] = int(occurrence[name])
    out["occurrence"] = occurrence
    if out.get("backend") is None:
        out["backend"] = ref_in["backend"]
    if out.get("uri") or out.get("path"):
        return out
    if ref_in.get("path"):
        folder = Path(ref_in["path"])
        if out.get("run") is None:
            if not folder.name.isdigit():
                raise ValueError(f"cannot derive an output run from {folder}: give imas_ref_out")
            out["run"] = int(folder.name) + 1
        out["path"] = str(folder.parent / str(int(out["run"])))
        return out
    if ref_in.get("uri"):
        raise ValueError("the input entry is given as a URI: give imas_ref_out (uri, path or run)")
    for key in ("shot", "user", "tokamak", "version"):
        if out.get(key) is None:
            out[key] = ref_in[key]
    if out.get("run") is None:
        out["run"] = int(ref_in["run"]) + 1
    return out


def ref_uri(ref):
    """IMAS URI of a reference: its uri, its path, or the legacy shot/run/user/... form."""
    if ref.get("uri"):
        return ref["uri"]
    if ref.get("path"):
        return f"imas:{ref['backend']}?path={Path(ref['path']).resolve().as_posix()}"
    return (f"imas:{ref['backend']}?user={ref['user']};pulse={int(ref['shot'])};"
            f"run={int(ref['run'])};database={ref['tokamak']};version={ref['version']}")


# ---------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------
def _array(node):
    """A leaf of an IDS as a float array (empty if it has no value)."""
    try:
        return np.asarray(node, dtype=float) if node.has_value else np.empty(0)
    except (AttributeError, TypeError, ValueError):
        return np.empty(0)


def _nonzero(*nodes):
    """The first of the leaves that holds a non-zero value."""
    for node in nodes:
        values = _array(node)
        if values.size and np.any(np.isfinite(values) & (values != 0)):
            return values
    return np.empty(0)


def synchronise(entry, time, occurrence):
    """Time of the last pellet launched at or before `time` (the first one if none), as
    hpi2_imas_interface.m: from time_slice[k].time, or from pellets.time when the pellets IDS
    has a homogeneous time base."""
    pellets = entry.get("pellets", occurrence, lazy=True, autoconvert=False)
    homogeneous = pellets.ids_properties.homogeneous_time == imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    if homogeneous:
        times = _array(pellets.time)
    else:
        times = np.array([float(time_slice.time) for time_slice in pellets.time_slice])
    if times.size == 0:
        raise ValueError(f"pellets occurrence {occurrence} holds no pellet")
    past = np.flatnonzero(times <= time)
    index = int(past.max()) if past.size else 0
    source = f"pellets.time[{index}]" if homogeneous else f"pellets.time_slice[{index}].time"
    print(f"Synchronisation: computing for the pellet launched at {times[index]:.6g} s "
          f"({source}) instead of the requested time {time:.6g} s")
    return float(times[index])


def last_time(entry, ids_name, occurrence):
    """Time of the last slice of an IDS occurrence in an entry (None if it holds none)."""
    try:
        ids = entry.get(ids_name, occurrence, lazy=True, autoconvert=False)
        if ids.ids_properties.homogeneous_time == imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS:
            times = _array(ids.time)
        elif ids_name == "pellets":
            times = np.array([float(time_slice.time) for time_slice in ids.time_slice])
        else:
            return None
    except Exception:   # the IDS is not in the entry yet
        return None
    return float(times.max()) if times.size else None


def hydrogen_label(pairs):
    """Hydrogen isotopes of (z_n, a) pairs, labelled as in hpi2_imas_interface.m."""
    found = {int(round(a)) for z_n, a in pairs if int(round(z_n)) == 1}
    if 2 in found:
        return "DT" if 3 in found else "D"
    if 3 in found:
        return "T"
    return "H" if 1 in found else ""


def pellet_radius(pellet):
    """Equivalent-sphere radius (m) from shape.type.index, as hpi2_imas_interface.m:
    1 sphere size[0]; 2 cylinder (3/4 size[0]^2 size[1])^(1/3); 3 general (3 prod(size)/4 pi)^(1/3)."""
    index = int(pellet.shape.type.index)
    size = _array(pellet.shape.size)
    if index == 1:
        return float(size[0])
    if index == 2:
        return float((0.75 * size[0] ** 2 * size[1]) ** (1.0 / 3.0))
    if index == 3:
        return float((3.0 * np.prod(size) / (4.0 * np.pi)) ** (1.0 / 3.0))
    raise ValueError(f"Unknown pellet shape (shape.type.index = {index})")


def read_inputs(uri, time, occurrence, pellet_rank):
    """Read the pellet and the equilibrium at `time` (closest samples, as HPI2) and the
    core_profiles at the last sample at or before `time`, so never a plasma after the pellet."""
    closest = imas.ids_defs.CLOSEST_INTERP
    previous = imas.ids_defs.PREVIOUS_INTERP
    with imas.DBEntry(uri, "r") as entry:
        if time < 0:   # the last pellet launched at or before abs(time), as HPI2
            time = synchronise(entry, abs(time), occurrence["pellets"])
        pellets = entry.get_slice("pellets", time, closest, occurrence["pellets"],
                                  autoconvert=False)
        if not len(pellets.time_slice) or len(pellets.time_slice[0].pellet) < pellet_rank:
            raise ValueError(f"no pellet {pellet_rank} in pellets occurrence "
                             f"{occurrence['pellets']} at {time} s")
        pellet = pellets.time_slice[0].pellet[pellet_rank - 1]
        geometry = pellet.path_geometry
        inputs = {
            "time": time,
            "t_pellet": float(pellets.time_slice[0].time),
            "velocity": float(pellet.velocity_initial),
            "radius": pellet_radius(pellet),
            "first_point": [float(geometry.first_point.r), float(geometry.first_point.z)],
            "second_point": [float(geometry.second_point.r), float(geometry.second_point.z)],
            "phi": [float(geometry.first_point.phi), float(geometry.second_point.phi)],
            "pellet_material": hydrogen_label((float(s.z_n), float(s.a)) for s in pellet.species),
        }

        # previous, not closest: the closest sample can lie after the pellet launch
        core = entry.get_slice("core_profiles", time, previous, occurrence["core_profiles"],
                               lazy=True, autoconvert=False)
        profiles = core.profiles_1d[0]
        core_time = _array(core.time)
        inputs["t_core_profiles"] = float(core_time[0]) if core_time.size else float(profiles.time)
        inputs["dd_version"] = imas.util.get_data_dictionary_version(core)
        inputs["rho"] = _array(profiles.grid.rho_tor_norm)
        inputs["ne"] = _nonzero(profiles.electrons.density_thermal, profiles.electrons.density)
        inputs["Te"] = _array(profiles.electrons.temperature)
        inputs["Ti"] = _array(profiles.t_i_average)
        inputs["q"] = _nonzero(profiles.q) if hasattr(profiles, "q") else np.empty(0)
        inputs["grid"] = {name: _array(getattr(profiles.grid, name))
                          for name in GRID_FIELDS if hasattr(profiles.grid, name)}
        inputs["plasma_type"] = hydrogen_label(
            (float(ion.element[0].z_n), float(ion.element[0].a))
            for ion in profiles.ion if len(ion.element))
        core_b0 = _array(core.vacuum_toroidal_field.b0)

        equilibrium = entry.get_slice("equilibrium", time, closest, occurrence["equilibrium"],
                                      lazy=True, autoconvert=False)
        time_slice = equilibrium.time_slice[0]
        b0 = _array(equilibrium.vacuum_toroidal_field.b0)
        if b0.size == 0:
            b0 = core_b0
        if b0.size == 0:
            raise ValueError("B0 found neither in equilibrium nor in core_profiles")
        inputs["B0"] = float(b0[0])
        if inputs["q"].size == 0:
            q_eq = _array(time_slice.profiles_1d.q)
            rho_eq = _array(time_slice.profiles_1d.rho_tor_norm)
            if q_eq.size == 0 or rho_eq.size != q_eq.size:
                raise ValueError("q found neither in core_profiles nor in equilibrium")
            inputs["q"] = np.interp(inputs["rho"], rho_eq, q_eq)
        outline = time_slice.boundary.outline
        inputs["boundary"] = (_array(outline.r), _array(outline.z))
        if inputs["boundary"][0].size < 3 and hasattr(time_slice, "boundary_separatrix"):
            outline = time_slice.boundary_separatrix.outline
            inputs["boundary"] = (_array(outline.r), _array(outline.z))

    for name in ("rho", "ne", "Te", "Ti"):
        if inputs[name].size == 0:
            raise ValueError(f"core_profiles has no {name} at {time} s")
    # q enters through its rational surfaces (1.5, 2, ...): the training data have q > 0
    inputs["q"] = np.abs(inputs["q"])
    return pellets, inputs


# ---------------------------------------------------------------------------------------
# geometry: where the pellet path enters the plasma
# ---------------------------------------------------------------------------------------
def plasma_entry_distance(first_point, second_point, r_boundary, z_boundary):
    """Distance (m) from first_point, along the line to second_point, at which the path
    crosses the plasma boundary inwards (0 if first_point is already inside), or None."""
    if r_boundary.size < 3:
        return None
    start = np.asarray(first_point, dtype=float)
    direction = np.asarray(second_point, dtype=float) - start
    direction /= np.linalg.norm(direction)
    a = np.column_stack([r_boundary, z_boundary])
    edge = np.roll(a, -1, axis=0) - a
    w = a - start
    denominator = direction[0] * edge[:, 1] - direction[1] * edge[:, 0]
    valid = np.abs(denominator) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * edge[:, 1] - w[:, 1] * edge[:, 0]) / denominator
        s = (w[:, 0] * direction[1] - w[:, 1] * direction[0]) / denominator
    crossing = valid & (s >= 0) & (s <= 1) & (t >= 0)
    if crossing.sum() % 2 == 1:        # odd number of crossings ahead: start is inside
        return 0.0
    return float(t[crossing].min()) if crossing.any() else None


# ---------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------
def code_info():
    """Name, commit, repository and version of the hpi2nn package."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=project_root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    try:
        from importlib.metadata import version
        package_version = version("hpi2nn")
    except Exception:   # not pip-installed: read it from pyproject.toml
        import re
        try:
            text = (project_root / "pyproject.toml").read_text(encoding="utf-8")
            package_version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
        except (OSError, AttributeError):
            package_version = ""
    return {"name": "hpi2nn", "commit": git("rev-parse", "HEAD"),
            "repository": git("remote", "get-url", "origin"), "version": package_version}


def parameters_xml(values):
    """The interface settings and the HPI2-NN inputs and outputs, as an XML string."""
    root = ET.Element("parameters")
    for key, value in values.items():
        ET.SubElement(root, key).text = " ".join(map(str, np.atleast_1d(value)))
    return ET.tostring(root, encoding="unicode")


def fill_code(ids, code, parameters, comment):
    ids.code.name = code["name"]
    ids.code.commit = code["commit"]
    ids.code.repository = code["repository"]
    ids.code.version = code["version"]
    ids.code.parameters = parameters
    ids.ids_properties.comment = comment
    ids.ids_properties.provider = getpass.getuser()
    ids.ids_properties.creation_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


# ---------------------------------------------------------------------------------------
# output IDSs
# ---------------------------------------------------------------------------------------
def pellets_output(pellets, pellet_rank, t_end, path):
    """The input pellets slice, dated at the end of ablation, with the ends of the path."""
    ids = copy.deepcopy(pellets)
    path_profiles = ids.time_slice[0].pellet[pellet_rank - 1].path_profiles
    for leaf in list(imas.util.tree_iter(path_profiles)):   # drop any earlier path
        leaf.value = np.empty(0)
    if path is not None:
        path_profiles.distance = path["distance"]
        path_profiles.position.r = path["r"]
        path_profiles.position.z = path["z"]
        path_profiles.position.phi = path["phi"]
        path_profiles.velocity = path["velocity"]
    ids.time_slice[0].time = t_end
    ids.time = np.array([t_end])
    return ids


def core_profiles_output(inputs, dne, dTe, t_end):
    """New core_profiles IDS on the input grid, with the profiles after the pellet."""
    ids = imas.IDSFactory(inputs["dd_version"]).new("core_profiles")
    ids.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    ids.time = np.array([t_end])
    ids.profiles_1d.resize(1)
    profiles = ids.profiles_1d[0]
    for name, values in inputs["grid"].items():
        if values.size:
            setattr(profiles.grid, name, values if values.ndim else float(values))
    ne_out = inputs["ne"] + dne
    Te_out = inputs["Te"] + dTe
    profiles.electrons.density = ne_out
    profiles.electrons.density_thermal = ne_out
    profiles.electrons.temperature = Te_out
    profiles.electrons.pressure = E_CHARGE * ne_out * Te_out
    profiles.electrons.pressure_thermal = E_CHARGE * ne_out * Te_out
    profiles.t_i_average = inputs["Ti"] * inputs["ne"] / ne_out
    profiles.time = t_end
    return ids


# ---------------------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------------------
def hpi2nn_imas_interface(time, imas_ref_in, imas_ref_out=None, pellet_rank=1, inj_value=None,
                          plot=False, write=True):
    """Run HPI2-NN on a pellet of an IMAS entry and write the result (see the module doc).

    Returns (pellets_out, core_profiles_out, result); `result` holds the HPI2-NN inputs and
    outputs (dne and dTe on the input rho grid, t_abl), the times and the references used."""
    ref_in = complete_ref_in(imas_ref_in)
    uri_in = ref_uri(ref_in)
    print(f"Reading IMAS data: {uri_in}  occurrences {ref_in['occurrence']}")
    pellets, inputs = read_inputs(uri_in, float(time), ref_in["occurrence"], int(pellet_rank))
    print(f"Pellet rank = {pellet_rank}; launched at {inputs['t_pellet']:.6g} s; "
          f"radius {inputs['radius'] * 1e3:.3g} mm; velocity {inputs['velocity']:.4g} m/s; "
          f"B0 {inputs['B0']:.4g} T; core_profiles sample at {inputs['t_core_profiles']:.6g} s")

    if inj_value is None:
        inj_value = find_closest_injection_line(inputs["first_point"], inputs["second_point"])
        if inj_value is None:
            raise ValueError(f"No HPI2-NN injection line matches the path "
                             f"{inputs['first_point']} -> {inputs['second_point']}")
    device = inj_value.split("_")[0]
    notes = []
    if ref_in.get("tokamak") and ref_in["tokamak"].upper() not in (device, "TEST"):
        notes.append(f"the entry is from {ref_in['tokamak']} but the path matches {inj_value}")
    if device in TRAINED_PELLET and inputs["pellet_material"] not in TRAINED_PELLET[device]:
        notes.append(f"pellet material '{inputs['pellet_material']}': HPI2-NN {device} was "
                     f"trained on {sorted(TRAINED_PELLET[device])} pellets only")
    if device in TRAINED_PLASMA and inputs["plasma_type"] not in TRAINED_PLASMA[device]:
        notes.append(f"plasma '{inputs['plasma_type']}': HPI2-NN {device} was trained on "
                     f"{sorted(TRAINED_PLASMA[device])} plasmas only")
    for note in notes:
        warnings.warn(note, stacklevel=2)

    # HPI2-NN, keeping its console output and warnings for the IDS comment
    log = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(log):
        warnings.simplefilter("always")
        dne, dTe, t_abl = evaluate_model(
            inputs["radius"], inputs["velocity"], inputs["rho"], inputs["Te"], inputs["ne"],
            inputs["Ti"], inputs["q"], inputs["B0"], inputs["first_point"],
            inputs["second_point"], inj_value=inj_value)
    print(log.getvalue(), end="")
    for warning in caught:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    notes += [str(warning.message) for warning in caught]

    # time and path: launch -> plasma edge (from the equilibrium boundary) -> full ablation
    entry_distance = plasma_entry_distance(inputs["first_point"], inputs["second_point"],
                                           *inputs["boundary"])
    velocity = inputs["velocity"]
    path = None
    if entry_distance is None:
        lead_in = 0.0
        notes.append("the plasma boundary was not found: the output time is the pellet time + "
                     "the in-plasma ablation time, without the flight to the plasma edge")
        warnings.warn(notes[-1], stacklevel=2)
    else:
        lead_in = entry_distance / velocity
        start = np.asarray(inputs["first_point"], dtype=float)
        line = np.asarray(inputs["second_point"], dtype=float) - start
        length = np.linalg.norm(line)
        distance = np.array([entry_distance, entry_distance + velocity * (t_abl - T_HOMOG)])
        r, z = (start[:, None] + line[:, None] / length * distance).tolist()
        phi = inputs["phi"][0] + (inputs["phi"][1] - inputs["phi"][0]) * distance / length
        path = {"distance": distance, "r": np.array(r), "z": np.array(z), "phi": phi,
                "velocity": np.full(2, velocity)}
    t_end = inputs["t_pellet"] + lead_in + t_abl

    code = code_info()
    parameters = parameters_xml({
        "time_requested": time, "time": inputs["time"],
        "time_core_profiles": inputs["t_core_profiles"], "pellet_rank": pellet_rank,
        "inj_value": inj_value, "pellet_radius": inputs["radius"], "velocity": velocity,
        "B0": inputs["B0"], "first_point": inputs["first_point"],
        "second_point": inputs["second_point"], "pellet_material": inputs["pellet_material"],
        "plasma_type": inputs["plasma_type"], "t_abl_in_plasma": t_abl,
        "t_flight_to_plasma": lead_in, "time_end_of_ablation": t_end,
    })
    comment = (f"HPI2-NN for IMAS (hpi2nn {code['version']} @ {code['commit'][:12]}): "
               f"pellet {pellet_rank} of pellets/{ref_in['occurrence']['pellets']} at "
               f"{inputs['t_pellet']:.6g} s, plasma from core_profiles/"
               f"{ref_in['occurrence']['core_profiles']} at {inputs['t_core_profiles']:.6g} s (last "
               f"sample at or before {inputs['time']:.6g} s) and equilibrium/"
               f"{ref_in['occurrence']['equilibrium']} at {inputs['time']:.6g} s, input entry "
               f"{uri_in}; injection line {inj_value}; in-plasma ablation time {t_abl * 1e3:.4g} ms"
               f" + flight to the plasma {lead_in * 1e3:.4g} ms."
               + "".join(f"\nWarning: {note}" for note in notes)
               + f"\nHPI2-NN log:\n{log.getvalue()}")

    pellets_out = pellets_output(pellets, int(pellet_rank), t_end, path)
    fill_code(pellets_out, code, parameters, comment)
    core_profiles_out = core_profiles_output(inputs, dne, dTe, t_end)
    fill_code(core_profiles_out, code, parameters, comment)

    ref_out = complete_ref_out(ref_in, imas_ref_out)
    uri_out = ref_uri(ref_out)
    if write:
        if uri_out == uri_in:
            raise ValueError(f"the output entry is the input entry ({uri_in}): its pellets and "
                             f"core_profiles would be overwritten; choose another run")
        print(f"Writing IMAS data: {uri_out}  occurrences {ref_out['occurrence']}")
        version = imas.util.get_data_dictionary_version(pellets_out)
        if imas.util.get_data_dictionary_version(core_profiles_out) != version:
            core_profiles_out = imas.convert_ids(core_profiles_out, version)
        with imas.DBEntry(uri_out, "a", dd_version=version) as entry:
            # put_slice appends: a slice at or before the last one would break the time order
            for name in ("pellets", "core_profiles"):
                last = last_time(entry, name, ref_out["occurrence"][name])
                if last is not None and t_end <= last:
                    raise ValueError(
                        f"{name}/{ref_out['occurrence'][name]} of {uri_out} already holds a slice "
                        f"at {last:.6g} s, not before this result ({t_end:.6g} s): write to another "
                        f"run or occurrence, or compute the pellets in time order")
            entry.put_slice(pellets_out, ref_out["occurrence"]["pellets"])
            entry.put_slice(core_profiles_out, ref_out["occurrence"]["core_profiles"])

    result = {**{k: v for k, v in inputs.items() if k not in ("grid", "boundary")},
              "inj_value": inj_value, "dne": dne, "dTe": dTe, "t_abl": t_abl,
              "t_flight_to_plasma": lead_in, "time_end_of_ablation": t_end,
              "uri_in": uri_in, "uri_out": uri_out if write else None, "warnings": notes}
    print(f"HPI2-NN: {inj_value}, peak dne {dne.max():.3e} m-3, in-plasma ablation time "
          f"{t_abl * 1e3:.4g} ms, output dated {t_end:.6g} s")
    if plot:
        plot_result(result)
    return pellets_out, core_profiles_out, result


def plot_result(result):
    """Profiles before and after the pellet (matplotlib is only needed here)."""
    import matplotlib.pyplot as plt

    rho = result["rho"]
    fig, (ax_ne, ax_te) = plt.subplots(1, 2, figsize=(10, 4))
    ax_ne.plot(rho, result["ne"], "k", label="before")
    ax_ne.plot(rho, result["ne"] + result["dne"], "r", label="after (HPI2-NN)")
    ax_ne.set(xlabel=r"$\rho_{tor,norm}$", ylabel=r"$n_e$ (m$^{-3}$)", title=result["inj_value"])
    ax_te.plot(rho, result["Te"], "k", label="before")
    ax_te.plot(rho, result["Te"] + result["dTe"], "r", label="after (HPI2-NN)")
    ax_te.set(xlabel=r"$\rho_{tor,norm}$", ylabel=r"$T_e$ (eV)",
              title=rf"$t_{{abl}}$ = {result['t_abl'] * 1e3:.3g} ms in the plasma")
    for ax in (ax_ne, ax_te):
        ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time", type=float, required=True,
                        help="time of the computation (s); negative: the last pellet before it")
    parser.add_argument("--uri", help="input entry as an IMAS URI")
    parser.add_argument("--path", help="input entry folder (holding master.h5)")
    parser.add_argument("--shot", type=int)
    parser.add_argument("--run", type=int)
    parser.add_argument("--user", help="IMAS user (default: the current user)")
    parser.add_argument("--tokamak", help="IMAS database name, e.g. west")
    parser.add_argument("--version", help="IMAS major version (default 3)")
    parser.add_argument("--backend", help="hdf5 (default), mdsplus, ...")
    parser.add_argument("--occ-equilibrium", type=int)
    parser.add_argument("--occ-core-profiles", type=int)
    parser.add_argument("--occ-pellets", type=int)
    parser.add_argument("--out-uri")
    parser.add_argument("--out-path")
    parser.add_argument("--out-shot", type=int)
    parser.add_argument("--out-run", type=int)
    parser.add_argument("--out-user")
    parser.add_argument("--out-tokamak")
    parser.add_argument("--out-occ-pellets", type=int)
    parser.add_argument("--out-occ-core-profiles", type=int)
    parser.add_argument("--pellet-rank", type=int, default=1)
    parser.add_argument("--inj-value", help="injection line, e.g. WEST_upHFS "
                                            "(default: detected from the pellet path)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="compute, but write nothing")
    args = parser.parse_args()

    imas_ref_in = {"uri": args.uri, "path": args.path, "shot": args.shot, "run": args.run,
                   "user": args.user, "tokamak": args.tokamak, "version": args.version,
                   "backend": args.backend,
                   "occurrence": {"equilibrium": args.occ_equilibrium,
                                  "core_profiles": args.occ_core_profiles,
                                  "pellets": args.occ_pellets}}
    imas_ref_out = {"uri": args.out_uri, "path": args.out_path, "shot": args.out_shot,
                    "run": args.out_run, "user": args.out_user, "tokamak": args.out_tokamak,
                    "occurrence": {"pellets": args.out_occ_pellets,
                                   "core_profiles": args.out_occ_core_profiles}}
    hpi2nn_imas_interface(args.time, imas_ref_in, imas_ref_out, args.pellet_rank,
                          args.inj_value, args.plot, not args.no_write)


if __name__ == "__main__":
    main()
