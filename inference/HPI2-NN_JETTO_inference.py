# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.
import numpy as np
from src.models.HPI2NN import evaluate_model

Data_0D=np.loadtxt('../JETTO_1DNOin.dat')
Te_in_dat=np.loadtxt('../JETTO_TEin.dat')
ne_in_dat=np.loadtxt('../JETTO_NEin.dat')
#For jetto multiply ne by 1e6 before calling the function
ne_in_dat=1e6*ne_in_dat
Ti_in_dat=np.loadtxt('../JETTO_TIin.dat')
q_in_dat=np.loadtxt('../JETTO_Qin.dat')
x_coord_dat=np.loadtxt('../JETTO_AMINin.dat')
x_coord_dat=x_coord_dat/x_coord_dat[-1]
B0=Data_0D[0]
pellet_radius=Data_0D[1]
vel_value=Data_0D[2]
#Reading injection first and second point from IDS
if Data_0D[3]==28:
    first_point=Data_0D[17:19]*1e-2
    second_point=Data_0D[19:21]*1e-2
    dne, dTe = evaluate_model(pellet_radius, vel_value, x_coord_dat, Te_in_dat, ne_in_dat, Ti_in_dat, q_in_dat, B0, first_point, second_point)
    
print(dne,dTe)
np.savetxt('../JETTO_DEPout.dat',dne/1e6)
np.savetxt('../JETTO_Tout.dat',[4.5228395e-04])#Ablation time
np.savetxt('../Outputready.nfo',[0])#Needed output for jetto implementation