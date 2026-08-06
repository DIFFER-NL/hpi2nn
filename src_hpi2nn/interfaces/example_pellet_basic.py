# Copyright 2024 DIFFER
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained TORAX example using the HPI2-NN pellet source.

The purpose of this example is to test that the coupling between TORAX and
HPI2-NN works and to show a minimal configuration. HPI2-NN is used as a model
implementation of the built-in TORAX ``pellet`` source (``model_name:
'hpi2_nn'``), registered through the TORAX registry pattern below.

Run with:

    run_torax --config=hpi2nn/src_hpi2nn/interfaces/example_pellet_basic.py

Requirements:
  - The 'hpi2nn' package installed (pip install -e <this repo>): it provides the
    surrogate model and its weights.
  - TORAX installed in the same environment.
"""

from torax import sources

from hpi2nn.src_hpi2nn.interfaces.torax_interface import HPI2NNPelletConfig

# Register HPI2-NN as the 'hpi2_nn' model of the built-in 'pellet' source. Must
# run before the CONFIG below is parsed into a ToraxConfig.
sources.register_source_model_config(HPI2NNPelletConfig, 'pellet')


CONFIG = {
    'profile_conditions': {},  # default profile conditions
    'plasma_composition': {},  # default plasma composition
    'numerics': {
        't_initial': 0.0,
        't_final': 10,
        'fixed_dt': 0.01,
        'min_dt': 1e-4,
        # The density equation must be evolved for the pellet to fuel the core.
        'evolve_density': True,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
    },
    # Circular geometry is only for testing and prototyping (no external files).
    'geometry': {
        'geometry_type': 'circular',
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        # Current source (for the psi equation).
        'generic_current': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'ei_exchange': {},
        'ohmic': {},
        # HPI2-NN pellet particle source (for the n_e equation): the 'hpi2_nn'
        # model implementation of the built-in 'pellet' source.
        'pellet': {
            'model_name': 'hpi2_nn',
            'trigger_times': [2.0, 6.0],
            'pellet_radius': 1.0e-3,  # [m]
            'pellet_velocity': 100.0,  # [m/s]
            'injection_line': 'WEST_upHFS',
            'use_model_ablation_time': False,
            'ablation_time': 1e-3,  # [s]
        },
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    # Mandatory for the HPI2-NN pellet source: the pellet-aware calculator aligns
    # time steps with the 'pellet' source's trigger times and ablation windows.
    'time_step_calculator': {
        'calculator_type': 'pellet_aware',
        'base_calculator': {'calculator_type': 'fixed'},
    },
}
