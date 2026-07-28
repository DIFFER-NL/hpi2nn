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
"""TORAX interface for the HPI2-NN pellet ablation surrogate.

This module implements HPI2-NN as an alternative model implementation of the
built-in TORAX 'pellet' source (whose default model is a Gaussian). It is
plugged into a TORAX config with the standard TORAX registry pattern, at the top
of the config module::

    from torax import sources
    from hpi2nn.src_hpi2nn.interfaces.torax_interface import HPI2NNPelletConfig

    sources.register_source_model_config(HPI2NNPelletConfig, 'pellet')

after which a TORAX config can select it under the 'pellet' source::

    'sources': {
        'pellet': {
            'model_name': 'hpi2_nn',
            'trigger_times': [2.0, 6.0],
            'pellet_radius': 1.0e-3,
            ...
        },
    },

The source returns the total HPI2-NN deposit profile as a step function in time,
active during the ablation window. The pellet-aware time step calculator
('time_step_calculator': {'calculator_type': 'pellet_aware'}) is required to align
time steps with the trigger and ablation windows.

Requirements:
  - The 'hpi2nn' package and its model weights (this repo, editable install).
  - TORAX installed in the same environment.
"""
import dataclasses
from typing import Annotated, Literal

import chex
import jax
import jax.numpy as jnp
import pydantic
import torax
from torax import sources as torax_sources
from torax._src import array_typing
from torax._src.geometry import geometry
from torax._src.sources import pellet_source
from torax._src.torax_pydantic import torax_pydantic

AllowedMachines = Literal['WEST', 'ITER', 'AUG']
AllowedInjectionLines = Literal[
    'WEST_upHFS',
    'WEST_midHFS',
    'WEST_lowHFS',
    'WEST_LFS',
    'ITER_upHFS',
    'AUG_upHFS',
]

MODEL_NAME: str = 'hpi2_nn'

# Get evaluate_model from HPI2NN to calculate dne and dTe
# hpi2nn is installed as an editable package (pip install -e), see its README.
def evaluate_hpi2nn_model(*args):
  from hpi2nn.src_hpi2nn.models.JAX_HPI2NN import evaluate_model  # pylint: disable=import-outside-toplevel

  return evaluate_model(*args)


# The whole ablation is one time step landing exactly on the trigger. The
# pellet-aware time-step calculator sets dt = ablation window, whether that
# window comes from the config ablation_time or the HPI2NN t_abl. The source
# is therefore active only at the trigger instant (within tol).

# If trigger times are used, check whether the current time is at a trigger
# instant and return the index of that trigger (if any).
def is_active_for_trigger_times(
    t: jax.Array,
    trigger_times: tuple[float, ...],
    tol: array_typing.FloatScalar,
) -> tuple[jax.Array, jax.Array]:
  """Returns (at_trigger, index) for an explicit list of trigger times."""
  tol = jnp.asarray(tol, dtype=t.dtype)
  triggers = jnp.asarray(trigger_times, dtype=t.dtype)
  at_trigger = jnp.abs(t - triggers) <= tol
  active = jnp.any(at_trigger)
  index = jnp.where(active, jnp.argmax(at_trigger), -1)
  return active, jnp.asarray(index, dtype=jnp.int32)

# If frequency is used, check whether the current time is at a periodic pellet
# instant (phase measured from t_start).
def is_active_for_frequency(
    t: jax.Array,
    frequency: array_typing.FloatScalar,
    frequency_t_start: array_typing.FloatScalar,
    tol: array_typing.FloatScalar,
) -> tuple[jax.Array, jax.Array]:
  """Returns (at_trigger, index=-1) for a periodic pellet.

  The phase is measured from frequency_t_start and no pellet fires at or before
  it.
  """
  frequency = jnp.asarray(frequency, dtype=t.dtype)
  positive_frequency = frequency > 0.0
  safe_frequency = jnp.where(
      positive_frequency, frequency, jnp.asarray(1.0, dtype=t.dtype)
  )
  period = 1.0 / safe_frequency
  tol = jnp.asarray(tol, dtype=t.dtype)
  t_start = jnp.asarray(frequency_t_start, dtype=t.dtype)
  # Phase is measured from t_start, no pellet fires at or before t_start.
  after_start = t > t_start + tol
  phase = jnp.mod(t - t_start + tol, period)
  # Float rounding can leave phase just below period instead of wrapping to 0
  # at a pellet time
  phase = jnp.where(
      period - phase < tol, jnp.asarray(0.0, dtype=t.dtype), phase
  )
  at_trigger = jnp.logical_and(
      jnp.logical_and(positive_frequency, after_start),
      phase <= tol,
  )
  return at_trigger, jnp.asarray(-1, dtype=jnp.int32)

# Call the right function to check if the pellet source should be active.
def is_pellet_active(source_params: 'RuntimeParams') -> tuple[jax.Array, jax.Array]:
  """Returns (at_trigger, index) for the HPI2-NN pellet runtime params."""
  t = jnp.asarray(source_params.current_time)
  if source_params.trigger_times is not None:
    return is_active_for_trigger_times(
        t=t,
        trigger_times=source_params.trigger_times,
        tol=source_params.trigger_tolerance,
    )
  if source_params.frequency is not None:
    return is_active_for_frequency(
        t=t,
        frequency=source_params.frequency,
        frequency_t_start=source_params.frequency_t_start,
        tol=source_params.trigger_tolerance,
    )
  return jnp.asarray(False), jnp.asarray(-1, dtype=jnp.int32)


# pylint: disable=invalid-name
# Runs the HPI2-NN model and returns the total deposit profile dne [m^-3] and the
# model-predicted ablation time t_abl [s]. The ablation time is the duration for
# which the pellet is considered active after being triggered.
def _run_hpi2nn_model(
    source_params: 'RuntimeParams',
    geo: torax.Geometry,
    core_profiles: torax.CoreProfiles,
    index_active_trigger: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  Te_eV = core_profiles.T_e.value * 1e3
  Ti_eV = core_profiles.T_i.value * 1e3
  q_cell = geometry.face_to_cell(core_profiles.q_face)
  rho_norm = geo.rho_norm

  if (
      source_params.pellet_radii is not None
      and source_params.pellet_velocities is not None
  ):
    safe_index = jnp.maximum(index_active_trigger, 0)
    pellet_radius = jnp.asarray(source_params.pellet_radii)[safe_index]
    pellet_velocity = jnp.asarray(source_params.pellet_velocities)[safe_index]
  else:
    pellet_radius = source_params.pellet_radius
    pellet_velocity = source_params.pellet_velocity

  dne, dTe, t_abl = evaluate_hpi2nn_model(
      pellet_radius, pellet_velocity, rho_norm, Te_eV,
      core_profiles.n_e.value, Ti_eV, q_cell, geo.B_0,
      source_params.injection_point_1, source_params.injection_point_2,
      source_params.injection_line,
  )
  return jnp.asarray(dne), t_abl


# The source profile is a step function in time; when active it is considered
# constant over the ablation window. HPI2-NN returns dne as total deposits so
# they are divided by the ablation time to get the source rate.
def calc_pellet_source(
    runtime_params: torax.RuntimeParams,
    geo: torax.Geometry,
    source_name: str,
    core_profiles: torax.CoreProfiles,
    calculated_source_profiles: torax_sources.SourceProfiles | None,
    unused_conductivity,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Calculates external source term for n_e from HPI2-NN pellets."""
  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, RuntimeParams)
  use = source_params.use_model_ablation_time

  is_active, index_active_trigger = is_pellet_active(source_params)

  def active_pellet(_):
    dne, t_abl = _run_hpi2nn_model(
        source_params, geo, core_profiles, index_active_trigger
    )
    divisor = t_abl if use else source_params.ablation_time
    return (dne / divisor,)

  def inactive_pellet(_):
    return (jnp.zeros_like(geo.rho_norm),)

  return jax.lax.cond(is_active, active_pellet, inactive_pellet, 0.0)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(torax_sources.RuntimeParams):
  """Runtime params for the HPI2-NN pellet model of the 'pellet' source."""

  pellet_radius: array_typing.FloatScalar
  pellet_velocity: array_typing.FloatScalar
  injection_line: str = dataclasses.field(metadata={'static': True})
  injection_point_1: tuple[float, float]
  injection_point_2: tuple[float, float]
  trigger_times: tuple[float, ...] | None
  # Per-trigger radius and velocity (must have same length as trigger_times).
  # If None, pellet_radius and pellet_velocity are used for all triggers.
  pellet_radii: tuple[float, ...] | None
  pellet_velocities: tuple[float, ...] | None
  frequency: array_typing.FloatScalar | None
  frequency_t_start: array_typing.FloatScalar
  ablation_time: array_typing.FloatScalar
  current_time: array_typing.FloatScalar
  # Must match the pellet-aware time step calculator's trigger_tolerance.
  trigger_tolerance: array_typing.FloatScalar
  # When True, the ablation window is the HPI2-NN-predicted t_abl instead of the
  # config ablation_time.
  use_model_ablation_time: bool = dataclasses.field(metadata={'static': True})

  # Generic hook read by the pellet-aware time step calculator. Returns
  # (at_trigger, ablation_time): the model-predicted t_abl at the trigger instant
  # (jax.lax.cond), otherwise the config ablation_time as a fallback. The model
  # runs only at the trigger instant.
  def ablation_step(
      self,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> tuple[jax.Array, jax.Array]:
    at_trigger, index = is_pellet_active(self)

    def get_hpi2nn_ablation_time(_):
      _dne, t_abl = _run_hpi2nn_model(self, geo, core_profiles, index)
      return jnp.asarray(t_abl)

    def fallback(_):
      return jnp.asarray(self.ablation_time)

    return at_trigger, jax.lax.cond(
        at_trigger, get_hpi2nn_ablation_time, fallback, 0.0
    )


class HPI2NNPelletConfig(torax_sources.SourceModelBase):
  """Configuration for the HPI2-NN pellet model of the 'pellet' source.

  Attributes:
    pellet_radius: Radius of the pellet in meters [m].
    pellet_velocity: Velocity of the pellet in meters per second [m/s].
    injection_line: Line of injection for the pellet.
      Allowed values are 'WEST_upHFS', 'WEST_midHFS', 'WEST_lowHFS',
      'WEST_LFS', 'ITER_upHFS', 'AUG_upHFS'.
    injection_point_1: First injection point (R, Z) [m].
      Not used if injection_line is specified.
    injection_point_2: Second injection point (R, Z) [m].
      Not used if injection_line is specified.
    trigger_times: A list of times to fire the pellet [s].
      If None, the pellet is not triggered by specific times.
    pellet_radii: A list of pellet radii corresponding to each trigger time [m].
      Must have the same length as trigger_times. If None, pellet_radius is
      used for all triggers.
    pellet_velocities: A list of pellet velocities corresponding to each
      trigger time [m/s]. Must have the same length as trigger_times. If None,
      pellet_velocity is used for all triggers.
    frequency: A frequency for firing the pellet [Hz].
      If 0 or None, the pellet is not triggered by frequency.
      If specified, the pellet is fired at regular intervals defined by the
      frequency.
    frequency_t_start: Reference time for frequency mode [s]. The phase is
      measured from this time and no pellet fires at or before it. Set it to
      t_initial to avoid a spurious first shot.
    use_model_ablation_time: When True (default), let HPI2-NN predict the ablation
      time t_abl for the deposit normalization and the time-step window.
    ablation_time: Let the user fix the ablation time [s] for all the pellets.
      This is used only when use_model_ablation_time is False.
      Shouldn't be used in most cases.
    trigger_tolerance: Time tolerance [s] used to decide whether the current time
      coincides with a pellet trigger. It must match the `trigger_tolerance` of
      the TORAX pellet-aware time step calculator (same default, 1e-8).
  """

  model_name: Annotated[Literal['hpi2_nn'], torax_pydantic.JAX_STATIC] = (
      'hpi2_nn'
  )

  pellet_radius: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.001)  # [m]
  )
  pellet_velocity: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(200.0)  # [m/s]
  )
  injection_line: Annotated[
      AllowedInjectionLines,
      torax_pydantic.JAX_STATIC,
  ] = 'WEST_upHFS'
  # The injection points are not used by the model but are kept in the config
  # because they are inputs of hpi2nn. They are not used if the injection line
  # is specified.
  injection_point_1: tuple[float, float] = (
      torax_pydantic.ValidatedDefault((1.8, 0.47))  # (R, Z) in meters
  )
  injection_point_2: tuple[float, float] = (
      torax_pydantic.ValidatedDefault((2.6192, -0.136))  # (R, Z) in meters
  )
  trigger_times: list[float] | None = None
  # Per-trigger radius and velocity (must have same length as trigger_times).
  # If None, pellet_radius and pellet_velocity are used for all triggers.
  pellet_radii: list[float] | None = None
  pellet_velocities: list[float] | None = None
  # OR a frequency, where 0 is "never" and value can be time-dependent.
  frequency: torax_pydantic.TimeVaryingScalar | None = None
  # Reference time for frequency mode: phase is measured from this time and no
  # pellet fires at or before it.
  frequency_t_start: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)  # [s]
  )
  ablation_time: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e-3)  # [s]
  )
  use_model_ablation_time: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  trigger_tolerance: float = torax_pydantic.ValidatedDefault(1e-8)

  mode: Annotated[
      torax_sources.Mode, torax_pydantic.JAX_STATIC
  ] = torax_sources.Mode.MODEL_BASED

  is_explicit: Annotated[bool, torax_pydantic.JAX_STATIC] = True

  @property
  def model_func(self) -> torax_sources.SourceProfileFunction:
    return calc_pellet_source

  @pydantic.model_validator(mode='after')
  def _validate_trigger_config(self):
    if self.trigger_times is not None and self.frequency is not None:
      raise ValueError(
          'Pellet source configuration must set either trigger_times or '
          'frequency, but not both.'
      )
    if self.frequency is not None and self.frequency.get_value(0.0) < 0.0:
      raise ValueError('frequency must be non-negative.')
    if self.frequency is not None and (
        self.pellet_radii is not None or self.pellet_velocities is not None
    ):
      raise ValueError(
          'pellet_radii and pellet_velocities are not supported with frequency '
          'mode; use pellet_radius and pellet_velocity (scalar) instead.'
      )
    if self.ablation_time.get_value(0.0) <= 0.0:
      raise ValueError('ablation_time must be strictly positive.')
    if self.trigger_times is not None and any(
        t < 0.0 for t in self.trigger_times
    ):
      raise ValueError('trigger_times must be non-negative.')
    if self.pellet_radii is not None and self.trigger_times is not None:
      if len(self.pellet_radii) != len(self.trigger_times):
        raise ValueError(
            f'pellet_radii length ({len(self.pellet_radii)}) must match '
            f'trigger_times length ({len(self.trigger_times)}).'
        )
    if self.pellet_velocities is not None and self.trigger_times is not None:
      if len(self.pellet_velocities) != len(self.trigger_times):
        raise ValueError(
            f'pellet_velocities length ({len(self.pellet_velocities)}) must '
            f'match trigger_times length ({len(self.trigger_times)}).'
        )
    return self

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    trigger_times = (
        tuple(self.trigger_times) if self.trigger_times is not None else None
    )
    return RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        pellet_radius=self.pellet_radius.get_value(t),
        pellet_velocity=self.pellet_velocity.get_value(t),
        injection_line=self.injection_line,
        injection_point_1=self.injection_point_1,
        injection_point_2=self.injection_point_2,
        trigger_times=trigger_times,
        pellet_radii=(
            tuple(self.pellet_radii) if self.pellet_radii is not None else None
        ),
        pellet_velocities=(
            tuple(self.pellet_velocities)
            if self.pellet_velocities is not None
            else None
        ),
        frequency=(
            self.frequency.get_value(t) if self.frequency is not None else None
        ),
        frequency_t_start=self.frequency_t_start.get_value(t),
        ablation_time=self.ablation_time.get_value(t),
        use_model_ablation_time=self.use_model_ablation_time,
        current_time=jnp.asarray(t),
        trigger_tolerance=self.trigger_tolerance,
    )

  def build_source(self) -> pellet_source.PelletSource:
    # HPI2-NN is a model implementation of the built-in 'pellet' source: it
    # reuses PelletSource with its own model function.
    return pellet_source.PelletSource(model_func=self.model_func)
