# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

#HPI2-NN model provides change in electron density due to pellet injection. Change in temperature is calculated through the adiabatic constraint.

import numpy as np
from scipy.optimize import curve_fit
import onnxruntime as ort
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Go two levels up: HPI2NN/
REPO_ROOT = THIS_DIR.parent.parent

# Path to the model weights
WEIGHTS_PATH = REPO_ROOT / "artifacts" / "models" 
SCALERS_PATH = REPO_ROOT / "artifacts" / "scalers"

injection_lines = {
	    "WEST_upperHFS": {"points": [(1.8, 0.47), (2.6192, -0.136)], "inj_value": 'WEST_upHFS'},
	    "WEST_HFS": {"points": [(1.8, 0), (3.38, 0)], "inj_value": 'WEST_midHFS'},
	    "WEST_X point": {"points": [(1.8, -0.33), (2.7336, -0.6884)], "inj_value": 'WEST_lowHFS'},
	    "WEST_LFS": {"points": [(3.38, 0.08), (1.8, 0.08)], "inj_value": 'WEST_LFS'},
        "ITER_upperHFS": {"points": [(3.96, 1.64), (4.65, 0.89)], "inj_value": 'ITER_upHFS'}
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
    #FOR JETTO multiply density by 1e6
    
    B0=-np.abs(B0)
    # Interpolate and scale new profile
    Te_interp = np.interp(np.linspace(0,1,101), x_coord, Te)
    ne_interp = np.interp(np.linspace(0,1,101), x_coord, ne)
    if inj_value==None:
        inj_value = find_closest_injection_line(first_point, second_point)

    print('Machine and angle detected as: ',inj_value)
    #LOADING THE WEIGHTS DEPENDING ON THE INJECTION AND MACHINE
    if inj_value=='WEST_upHFS':
        session = ort.InferenceSession(WEIGHTS_PATH / "WEST_upHFS.onnx")
    elif inj_value=='WEST_midHFS':
        session = ort.InferenceSession(WEIGHTS_PATH / "WEST_midHFS.onnx")
    elif inj_value=='WEST_lowHFS':
        session = ort.InferenceSession(WEIGHTS_PATH / "WEST_lowHFS.onnx")
    elif inj_value=='WEST_LFS':
        session = ort.InferenceSession(WEIGHTS_PATH / "WEST_LFS.onnx")
    elif inj_value=='ITER_upperHFS':
        session = ort.InferenceSession(WEIGHTS_PATH / "ITER_upHFS.onnx")
    else:
        raise ValueError("This is not a injection/Tokamak available in HPI2-NN")

    #Reading PCA profile dimensionality reduction parameters and normalization parameters
    if inj_value in ('WEST_upHFS', "WEST_midHFS.onnx","WEST_lowHFS.onnx","WEST_LFS.onnx"):
        data_Te = np.load(SCALERS_PATH / "WEST" / "pca_Te_data.npz")
        data_ne = np.load(SCALERS_PATH  / "WEST" / "pca_ne_data.npz")
        norm = np.load(SCALERS_PATH / "WEST" / "Normalization.npz")
        components_ne = data_ne["components"]
        
        #Sign switch for WEST last 2 components due to issue when generating PCA
        components_ne[1,:]=-components_ne[1,:]
        components_ne[2,:]=-components_ne[2,:]
    elif inj_value=="ITER_upperHFS":
        data_Te = np.load(SCALERS_PATH / "ITER" / "pca_Te_data.npz")
        data_ne = np.load(SCALERS_PATH / "ITER" / "pca_ne_data.npz")
        norm = np.load(SCALERS_PATH / "ITER" / "Normalization.npz")
        components_ne = data_ne["components"]
        
    components_Te = data_Te["components"]
    scaler_mean_Te = data_Te["scaler_mean"]
    scaler_std_Te = data_Te["scaler_std"]
    pca_mean_Te = data_Te["pca_mean"]
    scaler_mean_ne = data_ne["scaler_mean"]
    scaler_std_ne = data_ne["scaler_std"]
    pca_mean_ne = data_ne["pca_mean"]
    if inj_value in ('WEST_upHFS', "WEST_midHFS.onnx","WEST_lowHFS.onnx","WEST_LFS.onnx"):
        #again, for WEST was done different
        Te_scaled=Te_interp
        ne_scaled=ne_interp
    elif inj_value=="ITER_upperHFS":
        Te_scaled = (Te_interp - scaler_mean_Te) / scaler_std_Te
        ne_scaled = (ne_interp - scaler_mean_ne) / scaler_std_ne

    
    # Subtract PCA mean and apply projection to obtain 3 points per profile
    Te_centered = Te_scaled - pca_mean_Te
    Te_in_points = components_Te @ Te_centered  # shape (3,)
    
    ne_centered = ne_scaled - pca_mean_ne
    ne_in_points = components_ne @ ne_centered  # shape (3,)
    
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


    parameters=np.concatenate((ne_in_points,Te_in_points,params_Ti_Te,q_rat,np.array([B0]),params_inj))
    X=parameters
    print('NN parameters: ', parameters)
    
    scaler_X_mean=norm['scaler_X_mean']
    scaler_X_std= norm['scaler_X_std']
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
    ne_param=y[:6]
    # Te_param=y[6:]
    dne=1e19*two_gaussians(x_coord,*ne_param)
    # dTe=-1e2*two_gaussians(x_coord,*Te_param)

    #Adiabatic constraint to calculate Te
    Te_2=ne*Te/(ne+dne)
    dTe=Te_2-Te
    return dne, dTe
