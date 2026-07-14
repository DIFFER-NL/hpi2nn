# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

#HPI2-NN model provides change in electron density due to pellet injection. Change in temperature is calculated through the adiabatic constraint.

#New version of HPI2NN.py compatible with TORAX and JAX, using jaxxonnxruntime for interference.

#Dependencies (numpy, jax, onnx, onnxruntime, jaxonnxruntime) are installed automatically with `pip install -e .` (see pyproject.toml)

import numpy as np
import functools
import importlib
from pathlib import Path
import jax
import jax.numpy as jnp

THIS_DIR = Path(__file__).resolve().parent

# Go two levels up: HPI2NN/
REPO_ROOT = THIS_DIR.parent.parent

# Path to the model weights
WEIGHTS_PATH = REPO_ROOT / "artifacts_hpi2nn" / "models"
SCALERS_PATH = REPO_ROOT / "artifacts_hpi2nn" / "scalers"

injection_lines = {
	    "WEST_upperHFS": {"points": [(1.8, 0.47), (2.6192, -0.136)], "inj_value": 'WEST_upHFS'},
	    "WEST_HFS": {"points": [(1.8, 0), (3.38, 0)], "inj_value": 'WEST_midHFS'},
	    "WEST_X point": {"points": [(1.8, -0.33), (2.7336, -0.6884)], "inj_value": 'WEST_lowHFS'},
	    "WEST_LFS": {"points": [(3.38, 0.08), (1.8, 0.08)], "inj_value": 'WEST_LFS'},
        "ITER_upperHFS": {"points": [(3.96, 1.64), (4.65, 0.89)], "inj_value": 'ITER_upHFS'},
        "AUG_upperHFS": {"points": [(1.255, 0.915), (1.5954, -0.0253)], "inj_value": 'AUG_upHFS'}
	}


@functools.lru_cache(maxsize=8)
def get_onnx_infer_fn(model_path: str):
    """Loads a jaxonnxruntime callable from an ONNX file."""
    from jaxonnxruntime.backend import prepare
    import onnx

    model = onnx.load(model_path)
    rep = prepare(model)

    return lambda x: rep.run([x] if isinstance(x, list) else [x])


def _fit_exponential_ratio(x_coord, ti_over_te):
    """Fit y=a*exp(bx) with JAX-only LM, tuned to mimic scipy curve_fit behavior."""

    x = jnp.asarray(x_coord, dtype=jnp.float64)
    y = jnp.clip(jnp.asarray(ti_over_te, dtype=jnp.float64), 1e-12)

    # Normalize x for a better-conditioned optimization problem.
    x_shift = jnp.mean(x)
    x_scale = jnp.maximum(jnp.std(x), 1e-6)
    x_n = (x - x_shift) / x_scale

    # Linearized initialization: log(y) = c0 + c1*x_n.
    ones = jnp.ones_like(x_n)
    design = jnp.stack((ones, x_n), axis=1)
    rhs = jnp.log(y)
    normal = design.T @ design
    beta = jnp.linalg.solve(normal + 1e-12 * jnp.eye(2, dtype=x.dtype), design.T @ rhs)
    theta0 = jnp.asarray([beta[0], beta[1]], dtype=x.dtype)  # [log(a_n), b_n]

    def _loss(theta):
        log_a_n, b_n = theta
        pred = jnp.exp(log_a_n + b_n * x_n)
        residual = pred - y
        return 0.5 * jnp.sum(residual * residual)

    def _step(_, state):
        theta, lamb = state
        log_a_n, b_n = theta

        pred = jnp.exp(log_a_n + b_n * x_n)
        residual = pred - y

        # Jacobian wrt [log(a_n), b_n]
        jac = jnp.stack((pred, x_n * pred), axis=1)
        jtj = jac.T @ jac
        jtr = jac.T @ residual

        damp_matrix = lamb * jnp.diag(jnp.diag(jtj) + 1e-12)
        delta = jnp.linalg.solve(jtj + damp_matrix + 1e-12 * jnp.eye(2, dtype=x.dtype), jtr)
        theta_trial = theta - delta

        old_loss = _loss(theta)
        new_loss = _loss(theta_trial)
        accept = jnp.isfinite(new_loss) & (new_loss < old_loss)

        theta_next = jnp.where(accept, theta_trial, theta)
        lamb_next = jnp.where(accept, jnp.maximum(lamb * 0.3, 1e-12), jnp.minimum(lamb * 10.0, 1e12))
        return theta_next, lamb_next

    theta_final, _ = jax.lax.fori_loop(
        0,
        40,
        _step,
        (theta0, jnp.asarray(1e-3, dtype=x.dtype)),
    )

    log_a_n, b_n = theta_final
    a_n = jnp.exp(log_a_n)

    # Map back from normalized x_n to original x.
    b = b_n / x_scale
    a = a_n * jnp.exp(-b_n * x_shift / x_scale)

    params = jnp.asarray([a, b], dtype=x.dtype)
    return jnp.where(jnp.all(jnp.isfinite(params)), params, jnp.asarray([1.0, 1.0], dtype=x.dtype))


def calculate_angle(first_point, second_point):
    """Calculate the angle of the directed line segment w.r.t. the horizontal axis (counterclockwise)."""
    r1, z1 = first_point
    r2, z2 = second_point

    angle_rad = jnp.arctan2(z2 - z1, r2 - r1)  # Compute angle in radians
    angle_deg = jnp.degrees(angle_rad)  # Convert to degrees

        # Ensure angle is in the range [0, 360)
    angle_deg = jnp.where(angle_deg < 0, angle_deg + 360, angle_deg) #modif

    return angle_deg

def calculate_distance(first_point, second_point):
    """Calculate the Euclidean distance between two points."""
    return jnp.linalg.norm(jnp.asarray(first_point) - jnp.asarray(second_point))


def same_order_of_magnitude(x, y, tolerance=0.06):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    diff_zero = (x != 0) & (y != 0)
    mag_close = jnp.abs(jnp.log10(jnp.abs(x)) - jnp.log10(jnp.abs(y))) <= tolerance
    return jnp.where(diff_zero, mag_close, False) #if diff_zero=true then mag_close, else False


# Injection line -> ONNX weight file. WEST ships only the noBo (B0-dropped) models.
_MODEL_FILES = {
    'WEST_upHFS':  'WEST_upHFS_noBo_v4.onnx',
    'WEST_midHFS': 'WEST_midHFS_noBo_v4.onnx',
    'WEST_lowHFS': 'WEST_lowHFS_noBo_v4.onnx',
    'WEST_LFS':    'WEST_LFS_noBo_v4.onnx',
    'ITER_upHFS':  'ITER_upHFS_v4.onnx',
    'AUG_upHFS':   'AUG_upHFS_v4.onnx',
}
_WEST_LINES = ('WEST_upHFS', 'WEST_midHFS', 'WEST_lowHFS', 'WEST_LFS')


def _two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * jnp.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
            + a2 * jnp.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)))


def _scaler_paths(inj_value):
    """(scaler directory, normalization file) for a given injection line."""
    if inj_value in _WEST_LINES:
        return SCALERS_PATH / 'WEST', 'Normalization_v4.npz'
    if inj_value == 'ITER_upHFS':
        return SCALERS_PATH / 'ITER', 'Normalization_v4.npz'
    if inj_value == 'AUG_upHFS':
        return SCALERS_PATH / 'AUG', 'Normalization_AUG_A2_v3.npz'
    raise ValueError("This is not a injection/Tokamak available in HPI2-NN")


@functools.lru_cache(maxsize=8)
def _get_core(inj_value):
    """Build (once per injection line) a jit-compiled evaluator.

    All string-based model selection and weight/scaler loading happens here, on
    the host. The returned ``_core`` compiles into a single fused XLA kernel that
    includes the jaxonnxruntime forward pass and the Levenberg-Marquardt Ti/Te
    fit, so after the first (compiling) call every call is compiled inference
    only. Un-jitted, those two dominated the runtime (the LM fit alone was
    ~175 ms eager vs ~0.15 ms compiled).
    """
    if inj_value not in _MODEL_FILES:
        raise ValueError("This is not a injection/Tokamak available in HPI2-NN")

    onnx_path = (WEIGHTS_PATH / _MODEL_FILES[inj_value]).resolve()
    infer_fn = get_onnx_infer_fn(str(onnx_path))

    scaler_dir, norm_file = _scaler_paths(inj_value)
    data_Te = np.load(scaler_dir / 'pca_Te_data.npz')
    data_ne = np.load(scaler_dir / 'pca_ne_data.npz')
    norm = np.load(scaler_dir / norm_file)

    components_Te = jnp.asarray(data_Te['components'])
    scaler_mean_Te = jnp.asarray(data_Te['scaler_mean'])
    scaler_std_Te = jnp.asarray(data_Te['scaler_std'])
    pca_mean_Te = jnp.asarray(data_Te['pca_mean'])
    components_ne = jnp.asarray(data_ne['components'])
    scaler_mean_ne = jnp.asarray(data_ne['scaler_mean'])
    scaler_std_ne = jnp.asarray(data_ne['scaler_std'])
    pca_mean_ne = jnp.asarray(data_ne['pca_mean'])

    scaler_X_mean = norm['scaler_X_mean']
    scaler_X_std = norm['scaler_X_std']
    is_west = inj_value in _WEST_LINES
    if is_west:
        B0_IDX = len(scaler_X_mean) - 3   # B0 sits 3rd from the end (order-dependent!)
        scaler_X_mean = np.delete(scaler_X_mean, B0_IDX)
        scaler_X_std = np.delete(scaler_X_std, B0_IDX)
    scaler_X_mean = jnp.asarray(scaler_X_mean)
    scaler_X_std = jnp.asarray(scaler_X_std)
    scaler_y_mean = jnp.asarray(norm['scaler_y_mean'])
    scaler_y_std = jnp.asarray(norm['scaler_y_std'])

    @jax.jit
    def _core(pellet_radius, vel_value, x_coord, Te, ne, Ti, q, B0):
        # B0 is enforced negative (kept for the with-B0 machines; ignored for WEST noBo)
        B0 = -jnp.abs(B0)

        # Interpolate profiles onto the 101-point training grid, then scale.
        interp_grid = jnp.linspace(0, 1, 101)
        Te_interp = jnp.interp(interp_grid, x_coord, Te)
        ne_interp = jnp.interp(interp_grid, x_coord, ne)

        if is_west:
            out_of_range = jnp.logical_or(jnp.abs(B0) > 3.77, jnp.abs(B0) < 3.74)
            jax.lax.cond(
                out_of_range,
                lambda _: jax.debug.print(
                    'B0 is too far from the training range of WEST (-3.74 T, -3.77 T)'),
                lambda _: None,
                (),
            )

        Te_scaled = (Te_interp - scaler_mean_Te) / scaler_std_Te
        ne_scaled = (ne_interp - scaler_mean_ne) / scaler_std_ne

        # PCA-compress each profile to 3 coefficients.
        Te_in_points = components_Te @ (Te_scaled - pca_mean_Te)   # shape (3,)
        ne_in_points = components_ne @ (ne_scaled - pca_mean_ne)   # shape (3,)

        size_value = 4 / 3 * np.pi * pellet_radius ** 3            # pellet volume, m3
        params_inj = jnp.asarray([size_value, vel_value])

        # 3 rational/semi-rational q surfaces.
        q_targets = jnp.asarray([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=x_coord.dtype)
        q_candidates = jnp.interp(q_targets, q, x_coord)
        non_zero = q_candidates != 0
        n_candidates = q_candidates.shape[0]
        positions = jnp.where(non_zero, jnp.arange(n_candidates), n_candidates)
        top3_idx = jnp.sort(positions)[:3]
        q_rat = jnp.where(top3_idx < n_candidates, q_candidates[top3_idx], 1.0)

        # Exponential fit for Ti/Te (JAX-only implementation).
        params_Ti_Te = _fit_exponential_ratio(x_coord, Ti / Te)

        if is_west:   # no B0 for WEST (noBo models)
            parameters = jnp.concatenate(
                (ne_in_points, Te_in_points, params_Ti_Te, q_rat, params_inj))
        else:
            parameters = jnp.concatenate(
                (ne_in_points, Te_in_points, params_Ti_Te, q_rat, jnp.asarray([B0]), params_inj))

        X_norm = ((parameters.reshape(1, -1) - scaler_X_mean) / scaler_X_std).astype(jnp.float32)

        # Run inference using jaxonnxruntime (traced/fused under jit).
        y_norm = jnp.asarray(infer_fn(X_norm)).reshape(1, -1)
        y = (scaler_y_std * y_norm + scaler_y_mean).reshape(-1)   # unnormalize
        ne_param = y[:6]
        t_abl = y[6]
        dne = 1e19 * _two_gaussians(x_coord, *ne_param)

        # Make sure that dne is positive (print-only warnings; cannot raise under jit).
        neg_check = dne < 0
        neg_count = jnp.count_nonzero(neg_check)
        neg_fraction = neg_count / x_coord.shape[0]
        jax.lax.cond(
            neg_count > 0,
            lambda values: jax.debug.print(
                'Warning: HPI2-NN predicted negative dne in {count} positions out of {size}; values are clipped to 0.',
                count=values[0], size=values[1]),
            lambda _: None,
            (neg_count, x_coord.shape[0]),
        )
        jax.lax.cond(
            neg_fraction > 0.50,
            lambda frac: jax.debug.print(
                'Warning: HPI2-NN produced a large negative fraction before clipping: {frac_percent}%.',
                frac_percent=frac * 100),
            lambda _: None,
            neg_fraction,
        )
        dne = jnp.where(neg_check, 0, dne)

        # Adiabatic constraint to calculate Te.
        Te_2 = ne * Te / (ne + dne)
        dTe = Te_2 - Te

        return dne, dTe, t_abl

    return _core


def evaluate_model( pellet_radius, vel_value, x_coord, Te, ne, Ti, q, B0, first_point=[1.8, 0.47], second_point=[2.6192, -0.136],inj_value=None):
    # Pellet radius in m, velocity in m/s, Te and Ti in eV, ne in m-3, B0 in T, first point (R1,Z1) and second point (R2,Z2) in m
    #x coord preferred in rho_tor_norm, but using a_norm will not impact too much the result
    #B0 is suppose to be negative always (inforced anyway)
    #FOR JETTO multiply density by 1e6
    if inj_value is None:
        raise ValueError("In this version you need to provide the injection line to be able to load the correct model. Currently, giving 2 points to have the program to find an injection line does not work.")

    print('Machine and angle detected as: ', inj_value)

    # Build (or fetch cached) the jit-compiled evaluator for this injection line.
    core = _get_core(inj_value)

    # Coerce inputs to JAX arrays so the jitted core sees stable shapes/dtypes.
    x_coord = jnp.asarray(x_coord)
    return core(
        jnp.asarray(pellet_radius, dtype=x_coord.dtype),
        jnp.asarray(vel_value, dtype=x_coord.dtype),
        x_coord,
        jnp.asarray(Te), jnp.asarray(ne), jnp.asarray(Ti), jnp.asarray(q),
        jnp.asarray(B0, dtype=x_coord.dtype),
    )
