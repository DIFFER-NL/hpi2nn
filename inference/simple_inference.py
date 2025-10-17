# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.

#Run this script from HPI2NN/ as "python -m inference.simple_inference"
#FILL IN WITH YOUR OWN INPUT DATA
#Provided input data can give nonsense output
import numpy as np
from src.models.HPI2NN import evaluate_model


Te_in_dat=np.linspace(1000,100,100) #in eV
ne_in_dat=np.linspace(5*1e19,1e18,100) #in m-3

Ti_in_dat=Te_in_dat #in eV
q_in_dat=np.linspace(1,4,100) 
x_coord_dat=np.linspace(0,1,100) #coordinates in rho_tor_norm preferably
x_coord_dat=x_coord_dat/x_coord_dat[-1]
B0=-3 #in T
pellet_radius=0.001 #in m
vel_value=200 #in m/s


first_point=[1.8, 0.47] #[R,Z] in m
second_point= [2.6192, -0.136] #[R,Z] in m
dne, dTe = evaluate_model(pellet_radius, vel_value, x_coord_dat, Te_in_dat, ne_in_dat, Ti_in_dat, q_in_dat, B0, first_point, second_point)
    
print(dne,dTe)
np.savetxt('results/dne.txt',dne)#Store it in your own results folder

