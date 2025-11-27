#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math
import random

from vectors import ParacleteVec
from context import PhaseContext
from logos import (
    describe_axis,
    observer_profile,
    RealityScale,
    ObserverAxis,
    infer_reality_scale,
    soul_state_signature,
)
from temperature import compute_temperature


# ========== Section: Observer Slice ==========

@dataclass
class ObserverSlice:
    context: PhaseContext
    axis: ObserverAxis = ObserverAxis.SELF
    scale: RealityScale = RealityScale.ATOMIC
    resolution: float = 0.01
    attention_axes: List[int] = field(default_factory=lambda: list(range(12)))
    interpretation_map: Dict[int, str] = field(default_factory=dict)
    uncertainty: float = 0.001

    def __post_init__(self) -> None:
        if not self.interpretation_map:
            self.interpretation_map = self._default_interpretation_map()
        self._adjust_resolution_by_axis_and_scale()

    def _default_interpretation_map(self) -> Dict[int, str]:
        observable_names = [
            "mass",
            "charge",
            "spin",
            "momentum",
            "time",
            "frequency",
            "phase",
            "amplitude",
            "information",
            "entropy",
            "coherence",
            "meaning",
        ]
        mapping: Dict[int, str] = {}
        for i in range(12):
            if i < len(observable_names):
                mapping[i] = observable_names[i]
            else:
                axis_info = describe_axis(i)
                mapping[i] = axis_info["label"]
        return mapping

    def _axis_weight(self) -> float:
        if self.axis == ObserverAxis.SELF:
            return 1.0
        if self.axis == ObserverAxis.COLLECTIVE:
            return 0.9
        if self.axis == ObserverAxis.CREATION:
            return 0.8
        return 1.0

    def _scale_octave_index(self) -> int:
        return int(self.scale.value)

    def _adjust_resolution_by_axis_and_scale(self) -> None:
        base = self.resolution
        octave = self._scale_octave_index()
        ref = 5
        shift = octave - ref
        factor = 2.0 ** (abs(shift) / 4.0)
        if shift < 0:
            factor = 1.0 / factor
        axis_factor = self._axis_weight()
        self.resolution = max(1e-6, base * factor / axis_factor)
        self.uncertainty = max(1e-6, self.resolution / 10.0)

    def can_resolve(self, distance: float) -> bool:
        return distance >= self.resolution

    def filter_axes(self, vector: ParacleteVec) -> ParacleteVec:
        return vector.project_onto_axes(tuple(self.attention_axes))

    def add_uncertainty(self, vector: ParacleteVec) -> ParacleteVec:
        noise = ParacleteVec.from_list(
            [random.gauss(0.0, self.uncertainty) for _ in range(12)]
        )
        return vector + noise

    def scale_weight_for_state(self, vec: ParacleteVec) -> float:
        profile = observer_profile(vec)
        weights = profile["weights"]
        if self.axis == ObserverAxis.SELF:
            return float(weights.get("self", 0.0))
        if self.axis == ObserverAxis.COLLECTIVE:
            return float(weights.get("collective", 0.0))
        if self.axis == ObserverAxis.CREATION:
            return float(weights.get("creation", 0.0))
        return 0.0


# ========== Section: Projection and Interpretation ==========

def project_state_to_observer(state: ParacleteVec, observer: ObserverSlice) -> ParacleteVec:
    relative_state = state - observer.context.combined
    axis_weight = observer.scale_weight_for_state(relative_state)
    if axis_weight <= 0.0:
        scaled_state = relative_state * 0.0
    else:
        octave_idx = observer._scale_octave_index()
        ref = 5
        shift = octave_idx - ref
        octave_factor = 2.0 ** (shift / 4.0)
        scaled_state = relative_state * (axis_weight * octave_factor)
    filtered = observer.filter_axes(scaled_state)
    observed = observer.add_uncertainty(filtered)
    if observer.resolution > 0.0:
        data = list(observed.data)
        for i in range(len(data)):
            data[i] = round(data[i] / observer.resolution) * observer.resolution
        observed = ParacleteVec.from_list(data)
    return observed


def interpret_temperature_as_truth_confidence(
    temperature: float,
    threshold: float = 1.0,
    sigmoid_steepness: float = 1.0,
) -> float:
    return 1.0 / (1.0 + math.exp(-sigmoid_steepness * (temperature - threshold)))


def interpret_axes_as_observables(vector: ParacleteVec, observer: ObserverSlice) -> Dict[str, float]:
    observables: Dict[str, float] = {}
    for axis_idx in observer.attention_axes:
        if 0 <= axis_idx < 12:
            name = observer.interpretation_map.get(axis_idx, f"axis_{axis_idx}")
            observables[name] = vector.data[axis_idx]
    return observables


# ========== Section: Observer Network ==========

@dataclass
class ObserverNetwork:
    observers: List[ObserverSlice]
    consensus_threshold: float = 0.8

    def observe_state(self, state: ParacleteVec) -> List[ParacleteVec]:
        return [project_state_to_observer(state, obs) for obs in self.observers]

    def consensus_temperature(self, state: ParacleteVec) -> Tuple[float, float]:
        observations = self.observe_state(state)
        temps = [obs.norm() for obs in observations]
        if not temps:
            return 0.0, 0.0
        mean = sum(temps) / len(temps)
        variance = sum((t - mean) ** 2 for t in temps) / len(temps)
        std = math.sqrt(variance)
        return mean, std

    def consensus_vector(self, state: ParacleteVec) -> ParacleteVec:
        observations = self.observe_state(state)
        if not observations:
            return ParacleteVec.zeros()
        acc = ParacleteVec.zeros()
        for obs in observations:
            acc = acc + obs
        return (1.0 / len(observations)) * acc

    def agreement_score(self, state: ParacleteVec) -> float:
        observations = self.observe_state(state)
        if len(observations) < 2:
            return 1.0
        correlations: List[float] = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                a = observations[i]
                b = observations[j]
                na = a.norm()
                nb = b.norm()
                if na > 1e-10 and nb > 1e-10:
                    corr = a.dot(b) / (na * nb)
                    correlations.append(corr)
        if not correlations:
            return 1.0
        return (sum(correlations) / len(correlations) + 1.0) / 2.0

    def consensus_soul_state(self, state: ParacleteVec) -> Dict[str, Any]:
        consensus_vec = self.consensus_vector(state).normalized()
        signature = soul_state_signature(consensus_vec)
        temps, std = self.consensus_temperature(state)
        agreement = self.agreement_score(state)
        signature["consensus_temperature"] = temps
        signature["consensus_temperature_std"] = std
        signature["observer_agreement"] = agreement
        return signature


# ========== Section: Observer Construction ==========

def create_complementary_observers(context: PhaseContext) -> List[ObserverSlice]:
    observers: List[ObserverSlice] = []
    tangible_obs = ObserverSlice(
        context=context,
        axis=ObserverAxis.SELF,
        scale=RealityScale.ORGANISM,
        resolution=0.01,
        attention_axes=[0, 1, 2, 3],
        interpretation_map={
            0: "mass",
            1: "charge",
            2: "spin",
            3: "momentum",
        },
    )
    observers.append(tangible_obs)
    existential_obs = ObserverSlice(
        context=context,
        axis=ObserverAxis.COLLECTIVE,
        scale=RealityScale.BIOSPHERE,
        resolution=0.005,
        attention_axes=[4, 5, 6, 7],
        interpretation_map={
            4: "time",
            5: "frequency",
            6: "phase",
            7: "amplitude",
        },
    )
    observers.append(existential_obs)
    symbolic_obs = ObserverSlice(
        context=context,
        axis=ObserverAxis.CREATION,
        scale=RealityScale.ARCHETYPE,
        resolution=0.1,
        attention_axes=[8, 9, 10, 11],
        interpretation_map={
            8: "information",
            9: "entropy",
            10: "coherence",
            11: "meaning",
        },
    )
    observers.append(symbolic_obs)
    return observers


def create_hierarchical_observers(
    context: PhaseContext,
    resolutions: List[float] = None,
) -> List[ObserverSlice]:
    if resolutions is None:
        resolutions = [0.001, 0.01, 0.1, 1.0]
    scales = [
        RealityScale.ATOMIC,
        RealityScale.MICRO_ORGANIC,
        RealityScale.PLANETARY,
        RealityScale.INTERGALACTIC,
    ]
    axes = [
        ObserverAxis.SELF,
        ObserverAxis.COLLECTIVE,
        ObserverAxis.CREATION,
        ObserverAxis.CREATION,
    ]
    observers: List[ObserverSlice] = []
    for i, res in enumerate(resolutions):
        scale = scales[i] if i < len(scales) else RealityScale.SOURCE
        axis = axes[i] if i < len(axes) else ObserverAxis.SELF
        obs = ObserverSlice(
            context=context,
            axis=axis,
            scale=scale,
            resolution=res,
            attention_axes=list(range(12)),
        )
        observers.append(obs)
    return observers


# ========== Section: Observable Extractor ==========

class ObservableExtractor:
    def __init__(self, observer: ObserverSlice):
        self.observer = observer

    def extract_particle_properties(self, state: ParacleteVec) -> Dict[str, float]:
        observed = project_state_to_observer(state, self.observer)
        observables = interpret_axes_as_observables(observed, self.observer)
        mass = observables.get("mass", 0.0)
        charge = observables.get("charge", 0.0)
        spin = observables.get("spin", 0.0)
        momentum = observables.get("momentum", 0.0)
        energy = observables.get("energy", compute_temperature(observed))
        if abs(mass) > 1e-10:
            velocity = momentum / mass
        else:
            velocity = 0.0
        if velocity >= 1.0:
            rest_mass = mass
        else:
            rest_mass = mass / math.sqrt(1.0 + velocity * velocity)
        scale = infer_reality_scale(state)
        return {
            "mass": abs(mass),
            "charge": charge,
            "spin": abs(spin) % 2.0,
            "energy": abs(energy),
            "velocity": velocity,
            "rest_mass": rest_mass,
            "reality_scale": scale.name,
        }

    def extract_field_properties(self, state: ParacleteVec) -> Dict[str, float]:
        observed = project_state_to_observer(state, self.observer)
        observables = interpret_axes_as_observables(observed, self.observer)
        amplitude = observables.get("amplitude", 0.0)
        frequency = observables.get("frequency", 0.0)
        phase = observables.get("phase", 0.0)
        if abs(frequency) > 1e-10:
            wavelength = 1.0 / frequency
        else:
            wavelength = float("inf")
        wave_energy = amplitude * amplitude * abs(frequency)
        return {
            "amplitude": abs(amplitude),
            "frequency": abs(frequency),
            "phase": phase % (2.0 * math.pi),
            "wavelength": wavelength,
            "wave_energy": wave_energy,
            "intensity": amplitude * amplitude,
        }

    def extract_information_properties(self, state: ParacleteVec) -> Dict[str, float]:
        observed = project_state_to_observer(state, self.observer)
        observables = interpret_axes_as_observables(observed, self.observer)
        info = observables.get("information", 0.0)
        entropy = observables.get("entropy", 0.0)
        coherence = observables.get("coherence", 0.0)
        negentropy = max(0.0, info - entropy)
        if info > 0.0 and entropy > 0.0:
            complexity = info * entropy
        else:
            complexity = 0.0
        coh_abs = abs(coherence)
        coh_norm = coh_abs / (coh_abs + 1.0)
        organization = coh_norm * negentropy
        return {
            "information": abs(info),
            "entropy": abs(entropy),
            "coherence": coh_norm,
            "negentropy": negentropy,
            "complexity": complexity,
            "organization": organization,
        }
