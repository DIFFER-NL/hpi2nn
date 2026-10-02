# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

#HPI2-NN model provides change in electron density due to pellet injection. Change in temperature is calculated through the adiabatic constraint.

import json
import numpy as np
from scipy.optimize import curve_fit
import onnxruntime as ort
from pathlib import Path
import warnings

THIS_DIR = Path(__file__).resolve().parent

# Go two levels up: HPI2NN/
REPO_ROOT = THIS_DIR.parent.parent

# Path to the model weights
WEIGHTS_PATH = REPO_ROOT / "artifacts_hpi2nn" / "models"
SCALERS_PATH = REPO_ROOT / "artifacts_hpi2nn" / "scalers"

# Lower bound of the ablation time. t_abl is defined as (in-plasma path length)/velocity
# + 0.1 ms, so it can never be shorter than 0.1 ms; the network predicts it as a free
# output and can undershoot, typically for very hot plasmas or other extrapolations.
T_ABL_MIN = 1e-4  # s


class HPI2NNOutOfDomainWarning(UserWarning):
    """An input lies outside the range the model was trained on.

    evaluate_model issues it once per call, listing every quantity outside the training
    range of the injection line; the outputs are not changed. Silence it with
    warnings.simplefilter("ignore", HPI2NNOutOfDomainWarning), or make it an error with
    "error" instead of "ignore".
    """


# Training domain (artifacts_hpi2nn/scalers/<device>/training_domain.json): per injection
# line, the min and max over the rows the network was trained on of the physical
# quantities below and of the network inputs that describe the profile shapes. Profiles
# are read at these normalised radii, as in the paper's table of training ranges.
DOMAIN_RHO = (0.0, 0.5, 0.95)
# An input closer to an edge than this fraction of the range counts as inside, so that
# rounding (float32 in the JAX version) never flags a case from the training set itself.
DOMAIN_TOLERANCE = 1e-4
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
    values = {'v_pellet': float(vel_value), 'V_pellet': float(size_value),
              'abs_B0': float(np.abs(B0))}
    for x in DOMAIN_RHO:
        values[f'ne_{x:g}'] = float(np.interp(x, x_coord, ne))
        values[f'Te_{x:g}'] = float(np.interp(x, x_coord, Te))
    for x in (DOMAIN_RHO[0], DOMAIN_RHO[-1]):
        values[f'TiTe_{x:g}'] = float(np.interp(x, x_coord, Ti / Te))
    values['q95'] = float(np.interp(0.95, x_coord, q))
    return values


def out_of_domain(domain, inj_value, values):
    """Every value outside the training range of the line, as (key, value, min, max,
    distance beyond the edge in units of the range)."""
    found = []
    for key, (lo, hi) in domain['lines'][inj_value]['ranges'].items():
        if key not in values:
            continue
        width = hi - lo if hi > lo else max(abs(hi), 1e-300)
        distance = max(lo - values[key], values[key] - hi) / width
        if distance > DOMAIN_TOLERANCE:
            found.append((key, values[key], lo, hi, distance))
    return found


def domain_message(domain, inj_value, found):
    """The text of the out-of-domain warning."""
    inputs = domain['lines'][inj_value]['inputs']
    labels = {key: (entry['label'], entry['unit'], entry['scale'])
              for key, entry in domain['quantities'].items()}
    labels.update({key: (entry['label'], '', 1.0) for key, entry in domain['features'].items()})
    lines = [f'HPI2-NN input outside the training range of {inj_value}:']
    for key, value, lo, hi, distance in found:
        label, unit, scale = labels[key]
        unit = f' [{unit}]' if unit else ''
        note = ', not an input of this model' if key == 'abs_B0' and 'B0' not in inputs else ''
        lines.append(f'  {label}: {value * scale:.4g}, trained on {lo * scale:.4g} to '
                     f'{hi * scale:.4g}{unit} ({distance:.2f} of the range beyond{note})')
    lines.append(DOMAIN_ADVICE)
    return '\n'.join(lines)


injection_lines = {
	    "WEST_upperHFS": {"points": [(1.8, 0.47), (2.6192, -0.136)], "inj_value": 'WEST_upHFS'},
	    "WEST_HFS": {"points": [(1.8, 0), (3.38, 0)], "inj_value": 'WEST_midHFS'},
	    # WEST_lowHFS withdrawn 2026-09-23: 654 training cases on a path pinned to the lower X-point; the model gets the sign of the velocity, size and Te dependence wrong and is incoherent inside its own training range. Uncomment to restore.
	    # The geometry stays listed so that a lower-HFS injection is still identified
	    # and refused explicitly, rather than being matched to a neighbouring WEST line.
	    "WEST_X point": {"points": [(1.8, -0.33), (2.7336, -0.6884)], "inj_value": 'WEST_lowHFS', "withdrawn": True},
	    "WEST_LFS": {"points": [(3.38, 0.08), (1.8, 0.08)], "inj_value": 'WEST_LFS'},
        "ITER_upperHFS": {"points": [(3.96, 1.64), (4.65, 0.89)], "inj_value": 'ITER_upHFS'},
        "AUG_upperHFS": {"points": [(1.255, 0.915), (1.5954, -0.0253)], "inj_value": 'AUG_upHFS'}
	}

def calculate_angle(first_point, second_point):
    """Calculate the angle of the directed line segment w.r.t. the horizontal axis (counterclockwise)."""
    r1, z1 = first_point
    r2, z2 = second_point

    angle_rad = np.arctan2(z2 - z1, r2 - r1)  # Compute angle in radians
    angle_deg = np.degrees(angle_rad)  # Convert to degrees

        # Ensure angle is in the range [0, 360)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def calculate_distance(first_point, second_point):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(first_point) - np.array(second_point))

def same_order_of_magnitude(x, y, tolerance=0.06):
    if x == 0 or y == 0:
        return False
    return abs(np.log10(abs(x)) - np.log10(abs(y))) <= tolerance

def find_closest_injection_line(first_point, second_point):
    
    input_angle = calculate_angle(first_point, second_point)
    best_match = None
    min_score = float("inf")

    for label, data in injection_lines.items():
        ref_start, ref_end = data["points"]

        # Only compare R values (first coordinate of each point)
        R_input = [first_point[0], second_point[0]]
        R_ref = [ref_start[0], ref_end[0]]

        # Skip if R values are not in the same order of magnitude
        if not all(same_order_of_magnitude(ri, rr) for ri, rr in zip(R_input, R_ref)):
            continue

        ref_angle = calculate_angle(ref_start, ref_end)
        angle_diff = min(abs(input_angle - ref_angle), 360 - abs(input_angle - ref_angle))
        spatial_diff = (
            calculate_distance(first_point, ref_start) +
            calculate_distance(second_point, ref_end)
        ) / 2

        score = angle_diff + spatial_diff

        if score < min_score:
            min_score = score
            best_match = {"label": label, "inj_value": data["inj_value"]}

    return best_match['inj_value'] if best_match else None
def evaluate_model( pellet_radius, vel_value, x_coord, Te, ne, Ti, q, B0, first_point=[1.8, 0.47], second_point=[2.6192, -0.136],inj_value=None):
    # Pellet radius in m, velocity in m/s, Te and Ti in eV, ne in m-3, B0 in T, first point (R1,Z1) and second point (R2,Z2) in m
    #x coord preferred in rho_tor_norm, but using a_norm will not impact too much the result 
    #B0 is suppose to be negative always (inforced anyway)
    #FOR JETTO multiply density by 1e6 before getting it into this function
    
    B0=-np.abs(B0)
    # Interpolate and scale new profile
    Te_interp = np.interp(np.linspace(0,1,101), x_coord, Te)
    ne_interp = np.interp(np.linspace(0,1,101), x_coord, ne)
    if inj_value==None:
        inj_value = find_closest_injection_line(first_point, second_point)

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
 
    session = ort.InferenceSession(str(onnx_path))

    #Reading PCA profile dimensionality reduction parameters and normalization parameters
    if inj_value in ('WEST_upHFS', "WEST_midHFS","WEST_lowHFS","WEST_LFS"):
        data_Te = np.load(SCALERS_PATH / "WEST" / "pca_Te_data.npz")
        data_ne = np.load(SCALERS_PATH  / "WEST" / "pca_ne_data.npz")
        norm = np.load(SCALERS_PATH / "WEST" / "Normalization_v4.npz")
        components_ne = data_ne["components"]

    elif inj_value=="ITER_upHFS":
        data_Te = np.load(SCALERS_PATH / "ITER" / "pca_Te_data.npz")
        data_ne = np.load(SCALERS_PATH / "ITER" / "pca_ne_data.npz")
        norm = np.load(SCALERS_PATH / "ITER" / "Normalization_v4.npz")
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
    def expo(x,a,b):
        return a*np.exp(b*x)

		
		
    #inj_value 1==120, 2==180, 3==240, 4==0

    #Calculating pellet size
    size_value=4/3*np.pi*(pellet_radius)**3 #in m3
    params_inj=np.array([size_value,vel_value])
    
    #Calculating 3 rational and semirational surfaces of q profile
    q_rat=np.array([])
    for i in np.interp([1.5,2,2.5,3,3.5,4,4.5],q,x_coord):
        if i!=0:
            if q_rat.shape[0]>=3:
                break
            else:
                q_rat=np.append(q_rat,i)

    #Exponential fit for Ti/Te
    params_Ti_Te, covariance = curve_fit(expo, x_coord, Ti/Te, p0=[1,1])

    #Warn when the input is outside the training range of the line (B0 included for WEST)
    domain = load_training_domain(inj_value.split('_')[0])
    if domain is not None and inj_value in domain['lines']:
        values = domain_values(x_coord, Te, ne, Ti, q, B0, size_value, vel_value)
        values.update(zip(('ne_1', 'ne_2', 'ne_3'), ne_in_points))
        values.update(zip(('Te_1', 'Te_2', 'Te_3'), Te_in_points))
        values.update(zip(('Ti/Te_a', 'Ti/Te_b'), params_Ti_Te))
        values.update(zip(('q_rat_surf_1', 'q_rat_surf_2', 'q_rat_surf_3'), q_rat))
        found = out_of_domain(domain, inj_value, values)
        if found:
            warnings.warn(domain_message(domain, inj_value, found), HPI2NNOutOfDomainWarning,
                          stacklevel=2)

    if inj_value in ('WEST_upHFS', 'WEST_midHFS', 'WEST_lowHFS', 'WEST_LFS'): #no B0 for WEST
        parameters = np.concatenate((ne_in_points, Te_in_points, params_Ti_Te, q_rat, params_inj))
    elif inj_value=='AUG_upHFS':
    # AUG v5 (2026-09-29): 12 inputs. No B0 (it barely varies in the AUG database) and
    # no Ti/Te slope (Ti/Te is flat in every AUG plasma, so the slope is a constant);
    # as accurate as the 14-input v4 without its response to the Ti shape.
        parameters = np.concatenate((ne_in_points, Te_in_points, params_Ti_Te[:1], q_rat, params_inj))
    else:
        parameters = np.concatenate((ne_in_points, Te_in_points, params_Ti_Te, q_rat, np.array([B0]), params_inj))
    # parameters=np.concatenate((ne_in_points,Te_in_points,params_Ti_Te,q_rat,np.array([B0]),params_inj))
    X=parameters
    print('NN parameters: ', parameters)
    
    scaler_X_mean=norm['scaler_X_mean']
    scaler_X_std= norm['scaler_X_std']
    if inj_value in ('WEST_upHFS', 'WEST_midHFS', 'WEST_lowHFS', 'WEST_LFS'):
        B0_IDX = len(scaler_X_mean) - 3   # B0 sits 3rd from the end
        #Important: if order changes this is not true
        scaler_X_mean = np.delete(scaler_X_mean, B0_IDX)
        scaler_X_std  = np.delete(scaler_X_std,  B0_IDX)
    elif inj_value=='AUG_upHFS':
        # the stored statistics cover all 14 inputs: drop Ti/Te_b (8th) and B0 (3rd from the end)
        DROP_IDX = [7, len(scaler_X_mean) - 3]
        scaler_X_mean = np.delete(scaler_X_mean, DROP_IDX)
        scaler_X_std  = np.delete(scaler_X_std,  DROP_IDX)
    scaler_y_mean= norm['scaler_y_mean']
    scaler_y_std=norm['scaler_y_std']

    X_norm=(X.reshape(1,-1)-scaler_X_mean)/scaler_X_std

    def two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
        return (a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) + a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)))
	
    
    # Preparing inference with onnxruntime
    input_name = session.get_inputs()[0].name
    # input_shape = session.get_inputs()[0].shape
    
    
    # Run inference
    # print("Input shape:", X_norm.shape)
    X_norm=X_norm.astype(np.float32)
    y_norm = session.run(None, {input_name: X_norm}) #Evaluate NN
    y_norm=np.asarray(y_norm).reshape(1,-1)
    
    y=(scaler_y_std * y_norm + scaler_y_mean).reshape(-1)#Unnormalize
    # print('NN out: ',y)
    ne_param=y[:6]
    t_abl=y[6]
    # Te_param=y[6:]
    dne=1e19*two_gaussians(x_coord,*ne_param)
    # dTe=-1e2*two_gaussians(x_coord,*Te_param)

    #Make sure that t_abl respects the 0.1 ms floor of its definition
    if t_abl < T_ABL_MIN:
        warnings.warn(
            f'HPI2-NN predicted an ablation time of {t_abl*1e3:.3f} ms, below the 0.1 ms '
            f'floor of its definition; it has been clamped to 0.1 ms. The input is likely '
            f'outside the training range.',
            stacklevel=2,
        )
        t_abl = T_ABL_MIN

    #Make sure that dne is positive
    if np.count_nonzero(dne<0)>0:
        print(f'Warning: HPI2-NN has predicted negative dne, in {np.count_nonzero(dne < 0)} positions out of {x_coord.shape[0]}, it has been corrected to 0.')
        if np.count_nonzero(dne < 0)/x_coord.shape[0] >0.50:
            raise ValueError(f'Bad HPI2-NN prediction, {np.count_nonzero(dne < 0)/x_coord.shape[0]*100}% of positions were predicted negative, review input ranges.')
    dne=np.where(dne<0,0,dne)
    
    #Adiabatic constraint to calculate Te
    Te_2=ne*Te/(ne+dne)
    dTe=Te_2-Te
    

    return dne, dTe, t_abl
