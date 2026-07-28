.. _hpi2nn_torax_interface:

HPI2-NN pellet source (TORAX interface)
#######################################

`HPI2-NN <https://github.com/DIFFER-NL/hpi2nn>`_ is a machine-learning surrogate
of the HPI2 pellet ablation and deposition code, developed to accelerate
integrated modelling of pellet-fuelled tokamak discharges. This page documents
its interface with `TORAX <https://github.com/google-deepmind/torax>`_.

.. note::

   The ``hpi2nn`` package (the surrogate model and its weights) and TORAX must
   both be installed in the same environment.

Registering the model
=====================

HPI2-NN is exposed as an alternative model implementation of the built-in TORAX
``pellet`` source (whose default model is a Gaussian). It is plugged into a TORAX
config with the standard TORAX registry pattern, at the top of the config
module:

.. code-block:: python

    from torax import sources
    from hpi2nn.src_hpi2nn.interfaces.torax_interface import HPI2NNPelletConfig

    sources.register_source_model_config(HPI2NNPelletConfig, 'pellet')

The ``pellet`` source is then selected with ``model_name: 'hpi2_nn'``:

.. code-block:: python

    'sources': {
        'pellet': {
            'model_name': 'hpi2_nn',
            'injection_line': 'WEST_upHFS',
            'trigger_times': [2.0, 6.0],
            'pellet_radius': 1.0e-3,
            'pellet_velocity': 100.0,
        },
    },

The ``hpi2_nn`` model (``hpi2nn/src_hpi2nn/interfaces/torax_interface.py``) calls
HPI2-NN to obtain the particle deposition profile of a pellet with the
characteristics and at the time chosen by the user (trigger time), using the
``T_e``, ``T_i``, ``n_e`` and ``q`` profiles from TORAX:

.. code-block:: python

    dne, dTe, t_abl = evaluate_hpi2nn_model(
        radius, velocity, rho_norm, Te_eV, ne, Ti_eV, q_cell, B_0,
        injection_point_1, injection_point_2, injection_line,
    )

The magnetic field ``B_0`` is passed as an input but is no longer used inside
HPI2-NN in the latest version for WEST.

``injection_point_1`` and ``injection_point_2`` can be used by HPI2-NN to find
the closest matching injection line, but this feature is currently not exposed
through the interface (the injection line is selected directly via
``injection_line``).

See ``hpi2nn/src_hpi2nn/interfaces/torax_interface.py`` for the full list of
config attributes (pellet radius/velocity, per-trigger ``pellet_radii`` /
``pellet_velocities``, ``trigger_times`` or ``frequency``, ``injection_line``,
``ablation_time``, ``use_model_ablation_time``, ``trigger_tolerance``).

The source is explicit (is_explicit = True), so HPI2-NN is called once at each trigger time
and the deposit is held fixed during the implicit solve, instead of being re-evaluated
at every solver iteration.

Ablation time and source normalisation
=======================================

The pellet source is active during the *ablation time*, which represents the
time for the pellet to be fully ablated. Over this window the source is assumed
constant, with the total deposited density spread over the ablation time:

.. math::

    S_\mathrm{pellet} = \frac{\mathrm{d}n_e}{t_\mathrm{ablation}}

The ablation time is, by default, the value ``t_abl`` predicted by HPI2-NN
(``use_model_ablation_time = True``). Setting ``use_model_ablation_time =
False`` falls back to the user-provided ``ablation_time`` from the
configuration.

The pellet-aware time step calculator
(``torax/_src/time_step_calculator/pellet_aware_time_step_calculator.py``) is
required to ensure that the trigger times and the ablation window are resolved
exactly. For particle conservation it sets the time step over the ablation
window equal to the ablation time used for the source normalisation.

Advice and known issues
=======================

- Our tests showed that QLKNN produced better results than TGLF-NN during the
  post-pellet relaxation phase.
- Using a sawtooth model can sometimes create a temperature collapse during the
  relaxation phase. A pellet makes the pressure profile non-monotone, and a
  sawtooth crash on such a profile can drive a low-temperature collapse. To
  avoid this, you can suppress sawtooth crashes around each pellet with the generic
  ``suppression_times`` / ``suppression_duration`` fields of the sawtooth
  trigger model, setting ``suppression_times`` to the pellet ``trigger_times``:

  .. code-block:: python

      'mhd': {'sawtooth': {'trigger_model': {
          'model_name': 'simple',
          'suppression_times': [2.0, 6.0],   # = the pellet trigger_times
          'suppression_duration': 0.05,      # [s], 
      }}},
