# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

#HPI2-NN model provides change in electron density due to pellet injection. Change in temperature is calculated through the adiabatic constraint.

#New version of HPI2NN.py compatible with TORAX and JAX, using jaxxonnxruntime for interference.

#Dependencies (numpy, jax, onnx, onnxruntime, jaxonnxruntime) are installed automatically with `pip install -e .` (see pyproject.toml)

import json
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

# Lower bound of the ablation time (same as HPI2NN.py). t_abl is defined as
# (in-plasma path length)/velocity + 0.1 ms, so it can never be shorter than 0.1 ms.
T_ABL_MIN = 1e-4  # s

# Training domain, as in HPI2NN.py: per injection line, the min and max over the training
# rows of the physical quantities below and of the network inputs that describe the
# profile shapes (artifacts_hpi2nn/scalers/<device>/training_domain.json).
DOMAIN_RHO = (0.0, 0.5, 0.95)
DOMAIN_TOLERANCE = 1e-4   # fraction of the range: float32 rounding never flags a training case
DOMAIN_ADVICE = (
    'In held-out tests the error grew 1.2-1.5 times within a quarter of a range beyond the '
    'edge, and several-fold further out; pellet sizes and velocities outside the trained '
    'values gave errors of 13-34% of the peak.'
)


def load_training_domain(device):
    """The training domain of a device's models, or None if the artifacts have none."""
    path = SCALERS_PATH / device / "training_domain.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def domain_values(x_coord, Te, ne, Ti, q, B0, size_value, vel_value):
    """The physical quantities the training domain is stated in, for one input."""
    values = {'v_pellet': vel_value, 'V_pellet': size_value, 'abs_B0': jnp.abs(B0)}
    for x in DOMAIN_RHO:
        values[f'ne_{x:g}'] = jnp.interp(x, x_coord, ne)
        values[f'Te_{x:g}'] = jnp.interp(x, x_coord, Te)
    for x in (DOMAIN_RHO[0], DOMAIN_RHO[-1]):
        values[f'TiTe_{x:g}'] = jnp.interp(x, x_coord, Ti / Te)
    values['q95'] = jnp.interp(0.95, x_coord, q)
    return values


def print_out_of_domain(domain, inj_value, values):
    """HPI2NN.py's out-of-domain warning, printed with jax.debug.print so that it also
    works under jit: one line per quantity outside the training range, then the advice."""
    entry = domain['lines'][inj_value]
    labels = {key: (item['label'], item['unit'], item['scale'])
              for key, item in domain['quantities'].items()}
    labels.update({key: (item['label'], '', 1.0) for key, item in domain['features'].items()})
    any_outside = jnp.asarray(False)
    for key, (lo, hi) in entry['ranges'].items():
        if key not in values:
            continue
        label, unit, scale = labels[key]
        unit = f' [{unit}]' if unit else ''
        note = ', not an input of this model' if key == 'abs_B0' and 'B0' not in entry['inputs'] else ''
        width = hi - lo if hi > lo else max(abs(hi), 1e-300)
        distance = jnp.maximum(lo - values[key], values[key] - hi) / width
        outside = distance > DOMAIN_TOLERANCE
        any_outside = jnp.logical_or(any_outside, outside)
        message = ('Warning: HPI2-NN input outside the training range of ' + inj_value + ': '
                   + label + ': {value}, trained on ' + f'{lo * scale:.4g} to {hi * scale:.4g}'
                   + unit + ' ({distance} of the range beyond' + note + ')')
        _ = jax.lax.cond(
            outside,
            lambda args, message=message, scale=scale: jax.debug.print(
                message, value=args[0] * scale, distance=args[1]),
            lambda _: None,
            (values[key], distance),
        )
    _ = jax.lax.cond(any_outside, lambda _: jax.debug.print(DOMAIN_ADVICE), lambda _: None, ())


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

     
def evaluate_model( pellet_radius, vel_value, x_coord, Te, ne, Ti, q, B0, first_point=[1.8, 0.47], second_point=[2.6192, -0.136],inj_value=None):
    # Pellet radius in m, velocity in m/s, Te and Ti in eV, ne in m-3, B0 in T, first point (R1,Z1) and second point (R2,Z2) in m
    #x coord preferred in rho_tor_norm, but using a_norm will not impact too much the result 
    #B0 is suppose to be negative always (inforced anyway)
    #FOR JETTO multiply density by 1e6
    

    B0=-jnp.abs(B0)
    # Interpolate and scale new profile
    interp_grid = jnp.linspace(0,1,101)
    Te_interp = jnp.interp(interp_grid, x_coord, Te)
    ne_interp = jnp.interp(interp_grid, x_coord, ne)
    if inj_value==None:
        raise ValueError("In this version you need to provide the injection line to be able to load the correct model. Currently, giving 2 points to have the program to find an injection line does not work.")

    print('Machine and angle detected as: ',inj_value)
    #LOADING THE WEIGHTS DEPENDING ON THE INJECTION AND MACHINE
    if inj_value=='WEST_upHFS':
        onnx_path = (WEIGHTS_PATH / "WEST_upHFS_noBo_v4.onnx").resolve()
    elif inj_value=='WEST_midHFS':
        onnx_path = (WEIGHTS_PATH / "WEST_midHFS_noBo_v4.onnx").resolve()
    elif inj_value=='WEST_lowHFS':
        raise ValueError(
            "The WEST lower-HFS (X-point) model has been withdrawn: it is trained on "
            "654 cases and does not reproduce the sign of the velocity, pellet-size or "
            "Te dependence. Pass inj_value explicitly to use another line, or restore "
            "the commented branch below to re-enable it."
        )
        # onnx_path = (WEIGHTS_PATH / "WEST_lowHFS_noBo_v4.onnx").resolve()
    elif inj_value=='WEST_LFS':
        onnx_path = (WEIGHTS_PATH / "WEST_LFS_noBo_v4.onnx").resolve()
    elif inj_value=='ITER_upHFS':
        onnx_path = (WEIGHTS_PATH / "ITER_upHFS_v4.onnx").resolve()
    elif inj_value=='AUG_upHFS':
        onnx_path = (WEIGHTS_PATH / "AUG_upHFS_v5.onnx").resolve()
    else:
        raise ValueError("This is not a injection/Tokamak available in HPI2-NN")
 
 
    infer_fn = get_onnx_infer_fn(str(onnx_path))


    #Reading PCA profile dimensionality reduction pajnp.abs(ra) > 3.77, jnp.abs(B0)normalization parameters
    if inj_value in ('WEST_upHFS', "WEST_midHFS","WEST_lowHFS","WEST_LFS"):
        data_Te = jnp.load(SCALERS_PATH / "WEST" / "pca_Te_data.npz")
        data_ne = jnp.load(SCALERS_PATH  / "WEST" / "pca_ne_data.npz")
        norm = jnp.load(SCALERS_PATH / "WEST" / "Normalization_v4.npz")
        components_ne = data_ne["components"]

        # #Sign switch for WEST last 2 components due to issue when generating PCA
        # components_ne[1,:]=-components_ne[1,:]
        # components_ne[2,:]=-components_ne[2,:]
    elif inj_value=="ITER_upHFS":
        data_Te = jnp.load(SCALERS_PATH / "ITER" / "pca_Te_data.npz")
        data_ne = jnp.load(SCALERS_PATH / "ITER" / "pca_ne_data.npz")
        norm = jnp.load(SCALERS_PATH / "ITER" / "Normalization_v4.npz")
        components_ne = data_ne["components"]

    elif inj_value=="AUG_upHFS":
        data_Te = np.load(SCALERS_PATH / "AUG" / "pca_Te_data.npz")
        data_ne = np.load(SCALERS_PATH / "AUG" / "pca_ne_data.npz")
        norm = np.load(SCALERS_PATH / "AUG" / "Normalization_AUG_A2_v3.npz")
        components_ne = data_ne["components"]

    components_Te = data_Te["components"]
    scaler_mean_Te = data_Te["scaler_mean"]
    scaler_std_Te = data_Te["scaler_std"]
    pca_mean_Te = data_Te["pca_mean"]
    scaler_mean_ne = data_ne["scaler_mean"]
    scaler_std_ne = data_ne["scaler_std"]
    pca_mean_ne = data_ne["pca_mean"]
    
   #Normalization of Te and ne
    Te_scaled = (Te_interp - scaler_mean_Te) / scaler_std_Te
    ne_scaled = (ne_interp - scaler_mean_ne) / scaler_std_ne

    
    # Subtract PCA mean and apply projection to obtain 3 points per profile
    Te_centered = Te_scaled - pca_mean_Te
    Te_in_points = components_Te @ Te_centered  # shape (3,)
    
    ne_centered = ne_scaled - pca_mean_ne
    ne_in_points = components_ne @ ne_centered  # shape (3,)
    # print(Te_in_points, ne_in_points)
		
    size_value=4/3*np.pi*(pellet_radius)**3 #in m3
    params_inj=jnp.asarray([size_value,vel_value])
    
    #Calculating 3 rational and semirational surfaces of q profile
    q_targets = jnp.asarray([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=x_coord.dtype)
    q_candidates = jnp.interp(q_targets, q, x_coord)
    non_zero = q_candidates != 0
    n_candidates = q_candidates.shape[0]
    positions = jnp.where(non_zero, jnp.arange(n_candidates), n_candidates)
    top3_idx = jnp.sort(positions)[:3]
    q_rat = jnp.where(top3_idx < n_candidates, q_candidates[top3_idx], 1.0)
    #If there less than 3 valid coordinates, the missing coordinates are replaced with 1.0, but this likely indicates a problem with the configuration being used
   
    # Exponential fit for Ti/Te (JAX-only implementation)
    params_Ti_Te = _fit_exponential_ratio(x_coord, Ti / Te)

    # Print a warning when the input is outside the training range of the line (as HPI2NN.py)
    domain = load_training_domain(inj_value.split('_')[0])
    if domain is not None and inj_value in domain['lines']:
        values = domain_values(x_coord, Te, ne, Ti, q, B0, size_value, vel_value)
        values.update(zip(('ne_1', 'ne_2', 'ne_3'), ne_in_points))
        values.update(zip(('Te_1', 'Te_2', 'Te_3'), Te_in_points))
        values.update(zip(('Ti/Te_a', 'Ti/Te_b'), params_Ti_Te))
        values.update(zip(('q_rat_surf_1', 'q_rat_surf_2', 'q_rat_surf_3'), q_rat))
        print_out_of_domain(domain, inj_value, values)

    if inj_value in ('WEST_upHFS', 'WEST_midHFS', 'WEST_lowHFS', 'WEST_LFS'): #no B0 for WEST
        parameters = jnp.concatenate((ne_in_points, Te_in_points, params_Ti_Te, q_rat, params_inj))
    elif inj_value=='AUG_upHFS':
    # AUG v5 (2026-09-29): 12 inputs. No B0 (it barely varies in the AUG database) and
    # no Ti/Te slope (Ti/Te is flat in every AUG plasma, so the slope is a constant);
    # as accurate as the 14-input v4 without its response to the Ti shape.
        parameters = jnp.concatenate((ne_in_points, Te_in_points, params_Ti_Te[:1], q_rat, params_inj))
    else:
        parameters = jnp.concatenate((ne_in_points, Te_in_points, params_Ti_Te, q_rat, jnp.asarray([B0]), params_inj))
    #parameters=jnp.concatenate((ne_in_points,Te_in_points,params_Ti_Te,q_rat,jnp.asarray([B0]),params_inj))
    X=parameters
    #print('NN parameters: ', parameters)
    
    scaler_X_mean=norm['scaler_X_mean']
    scaler_X_std= norm['scaler_X_std']
    if inj_value in ('WEST_upHFS', 'WEST_midHFS', 'WEST_lowHFS', 'WEST_LFS'):
        B0_IDX = len(scaler_X_mean) - 3   # B0 sits 3rd from the end
        #Important: if order changes this is not true
        scaler_X_mean = jnp.delete(scaler_X_mean, B0_IDX)
        scaler_X_std  = jnp.delete(scaler_X_std,  B0_IDX)
    elif inj_value=='AUG_upHFS':
        # the stored statistics cover all 14 inputs: drop Ti/Te_b (8th) and B0 (3rd from the end)
        DROP_IDX = jnp.asarray([7, len(scaler_X_mean) - 3])
        scaler_X_mean = jnp.delete(scaler_X_mean, DROP_IDX)
        scaler_X_std  = jnp.delete(scaler_X_std,  DROP_IDX)
    scaler_y_mean= norm['scaler_y_mean']
    scaler_y_std=norm['scaler_y_std']

    X_norm=(X.reshape(1,-1)-scaler_X_mean)/scaler_X_std

    def two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
        return (a1 * jnp.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) + a2 * jnp.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)))
	

    # Run inference using jaxonnxruntime
    X_norm = jnp.asarray(X_norm, dtype=jnp.float32)
    JAX_output = infer_fn(X_norm)
    y_norm = jnp.asarray(JAX_output)  
    y_norm = y_norm.reshape(1, -1)
    

    y=(scaler_y_std * y_norm + scaler_y_mean).reshape(-1)#Unnormalize
    ne_param=y[:6]
    t_abl=y[6]
    # Te_param=y[6:]
    dne=1e19*two_gaussians(x_coord,*ne_param)
    # dTe=-1e2*two_gaussians(x_coord,*Te_param)

    # Make sure that t_abl respects the 0.1 ms floor of its definition
    _ = jax.lax.cond(
        t_abl < T_ABL_MIN,
        lambda value: jax.debug.print(
            'Warning: HPI2-NN predicted an ablation time of {t_ms} ms, below the 0.1 ms '
            'floor of its definition; it has been clamped to 0.1 ms. The input is likely '
            'outside the training range.',
            t_ms=value * 1e3,
        ),
        lambda _: None,
        t_abl,
    )
    t_abl = jnp.maximum(t_abl, T_ABL_MIN)

    # Make sure that dne is positive
    neg_check = dne < 0
    neg_count = jnp.count_nonzero(neg_check)
    neg_fraction = neg_count / x_coord.shape[0]

    _ = jax.lax.cond(
        neg_count > 0,
        lambda values: jax.debug.print(
            'Warning: HPI2-NN predicted negative dne in {count} positions out of {size}; values are clipped to 0.',
            count=values[0],
            size=values[1],
        ),
        lambda _: None,
        (neg_count, x_coord.shape[0]),
    )

    _ = jax.lax.cond(
        neg_fraction > 0.50,
        lambda frac: jax.debug.print(
            'Warning: HPI2-NN produced a large negative fraction before clipping: {frac_percent}%.',
            frac_percent=frac * 100,
        ),
        lambda _: None,
        neg_fraction,
    )

    dne = jnp.where(neg_check, 0, dne)

    #Adiabatic constraint to calculate Te
    Te_2=ne*Te/(ne+dne)
    dTe=Te_2-Te
    

    return dne, dTe, t_abl
