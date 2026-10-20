# Copyright (c) 2025 Dutch Institute for Fundamental Energy Research
# Licensed under the MIT License. See LICENSE file for details.
#Run HPI2-NN on a pellet stored in an IMAS entry and write the result to another entry
#(src_hpi2nn/interfaces/imas_interface.py, the HPI2-NN counterpart of HPI2's hpi2_imas_interface.m)
#Needs IMAS-Python and the IMAS access layer: pip install -e ".[imas]"
#FILL IN THE FIELDS BELOW WITH YOUR OWN ENTRY, then run: python inference_hpi2nn/HPI2-NN_IMAS_inference.py
import sys
import os
# 1. Get the path of the current script's directory (inference_hpi2nn/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path of the project root (.. one level up from inference_hpi2nn/)
project_root = os.path.join(current_dir, os.pardir)

# 3. Add the project root to the system path
sys.path.insert(0, project_root)
from src_hpi2nn.interfaces.imas_interface import hpi2nn_imas_interface

time = -37.5  #in s; negative: the last pellet launched at or before abs(time), from pellets.time_slice[k].time

#Input entry: shot/run/tokamak are looked up in the public folder of user, ~user/public/imasdb/tokamak/version/shot/run
#path (the folder of the run, holding master.h5) or uri (any IMAS URI) replace them when given
imas_ref_in = {
    'shot': 54304,
    'run': 0,
    'tokamak': 'west',     #IMAS database name
    'user': None,          #None: you
    'version': '3',        #IMAS major version
    'backend': 'hdf5',     #'hdf5' or 'mdsplus'
    'path': None,          #e.g. '/home/user/public/imasdb/west/3/54304/0'
    'uri': None,           #e.g. 'imas:hdf5?path=/home/user/public/imasdb/west/3/54304/0'
    'occurrence': {'equilibrium': 0, 'core_profiles': 0, 'pellets': 0},
}

#Output entry: a field left at None is taken from the input entry, the run from the input run + 1
#It must not be the input entry, and its slices are added in time order (compute the pellets in time order)
imas_ref_out = {
    'shot': None,
    'run': 1,
    'tokamak': None,
    'user': None,
    'path': None,          #with an input path: None gives the folder of the output run next to the input one
    'uri': None,
    'occurrence': {'core_profiles': 0, 'pellets': 1},
}

pellet_rank = 1     #which pellet of the time slice (1 = first, as in HPI2)
inj_value = None    #None: the injection line is detected from the pellet path; or e.g. 'WEST_upHFS'

pellets_out, core_profiles_out, result = hpi2nn_imas_interface(time, imas_ref_in, imas_ref_out, pellet_rank, inj_value=inj_value)

print('Change in density (m-3)\n', result['dne'], '\nChange in temperature (eV)\n', result['dTe'])
print('Ablation time in the plasma (s)', result['t_abl'], '\nOutput written at (s)', result['time_end_of_ablation'], 'in', result['uri_out'])
