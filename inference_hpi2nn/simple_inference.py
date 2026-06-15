# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

#Run this script from HPI2NN/ as "python -m inference.simple_inference"
#FILL IN WITH YOUR OWN INPUT DATA or read the test data
#Provided input data can give nonsense output
import sys
import os
import numpy as np
import pandas as pd
# 1. Get the path of the current script's directory (inference/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path of the project root (.. one level up from inference/)
project_root = os.path.join(current_dir, os.pardir)

# 3. Add the project root to the system path
sys.path.insert(0, project_root)
from src_hpi2nn.models.HPI2NN import evaluate_model
csv_path = os.path.join(project_root, 'results_hpi2nn/Test_data_hpi2nn.csv')
df=pd.read_csv(csv_path)
Te_in_dat=df['Te'].values #in eV
ne_in_dat=df['ne'].values#in m-3
x_coord_dat=df['rho'].values #coordinates in rho_tor_norm preferably
x_coord_dat=x_coord_dat/x_coord_dat[-1]
Ti_in_dat=df['Ti'].values #in eV
q_in_dat=df['q'].values

B0=-3.76 #in T #For WEST, between -3.74 T and -3.76 T
pellet_radius=0.001 #in m
vel_value=200 #in m/s

first_point=[1.8, 0.47] #[R,Z] in m
second_point= [2.6192, -0.136] #[R,Z] in m
print(Te_in_dat,ne_in_dat, Ti_in_dat)
# inj_val='ITER_upHFS'

dne, dTe, t_abl= evaluate_model(pellet_radius, vel_value, x_coord_dat, Te_in_dat, ne_in_dat, Ti_in_dat, q_in_dat, B0, first_point, second_point,inj_value=inj_val)
    
print('Change in density (m-3)\n',dne,'Change in temperature (eV)\n', dTe, 'Ablation time (s)\n', t_abl)
np.savetxt(os.path.join(project_root, 'results_hpi2nn/dne.txt'),dne)#Store it in your own results folder

