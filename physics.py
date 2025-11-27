"""
Physical Interpretation Layer

This module provides abstract mappings from 12D Paraclete states to
physical concepts like fields, particles, and creation hierarchies.
This is an extensible framework for interpreting the mathematical
structure in physical terms.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

from vectors import ParacleteVec
from logos import get_modality_axes
from temperature import (
    compute_temperature,
    decompose_temperature_by_logos,
    classify_temperature_regime,
)


@dataclass
class FieldState:
    position_12d: ParacleteVec
    time: float
    tags: Dict[str, Any] = field(default_factory=dict)
    field_strength: float = 0.0
    potential: float = 0.0

    @property
    def temperature(self) -> float:
        return compute_temperature(self.position_12d)

    def compute_field_strength(self) -> float:
        self.field_strength = self.temperature
        return self.field_strength

    def compute_potential(self) -> float:
        T = self.temperature
        self.potential = math.log1p(T) if T > 0.0 else 0.0
        return self.potential


@dataclass
class ParticleLike:
    name: str
    state: FieldState
    stability_score: float
    mass_like: float = 1.0
    charge_like: float = 0.0
    spin_like: float = 0.0

    def is_stable(self) -> bool:
        return self.stability_score > 0.5

    def binding_energy(self) -> float:
        return self.stability_score * self.state.temperature


@dataclass
class CreationLayer:
    name: str
    elements: List[CreationLayer | ParticleLike] = field(default_factory=list)
    descriptor: Dict[str, Any] = field(default_factory=dict)
    scale: float = 1.0

    def add_element(self, element: CreationLayer | ParticleLike) -> None:
        self.elements.append(element)

    def count_particles(self) -> int:
        count = 0
        for elem in self.elements:
            if isinstance(elem, ParticleLike):
                count += 1
            elif isinstance(elem, CreationLayer):
                count += elem.count_particles()
        return count


def map_paraclete_to_em(field_state: FieldState) -> Dict[str, float]:
    pos = field_state.position_12d
    tangible_axes = get_modality_axes("tangible")
    electric_projection = pos.project_onto_axes(tuple(tangible_axes))
    electric_field = electric_projection.norm()
    existential_axes = get_modality_axes("existential")
    magnetic_projection = pos.project_onto_axes(tuple(existential_axes))
    magnetic_field = magnetic_projection.norm()
    em_energy = 0.5 * (electric_field ** 2 + magnetic_field ** 2)
    poynting = electric_field * magnetic_field
    field_ratio = electric_field / magnetic_field if magnetic_field > 0.0 else float("inf")
    return {
        "electric_field": electric_field,
        "magnetic_field": magnetic_field,
        "em_energy": em_energy,
        "poynting_flux": poynting,
        "field_ratio": field_ratio,
    }


def map_paraclete_to_gravity(field_state: FieldState) -> Dict[str, float]:
    pos = field_state.position_12d
    logos_temps = decompose_temperature_by_logos(pos)
    mass_indicator = logos_temps.get("tangible", 0.0)
    curvature_indicator = logos_temps.get("existential", 0.0)
    r_s = 2.0 * mass_indicator
    r = field_state.temperature
    potential = -mass_indicator / r if r > 0.0 else 0.0
    tidal = curvature_indicator / (r ** 2) if r > 0.0 else 0.0
    return {
        "mass_indicator": mass_indicator,
        "curvature_indicator": curvature_indicator,
        "schwarzschild_radius": r_s,
        "gravitational_potential": potential,
        "tidal_force": tidal,
    }


def map_paraclete_to_nuclear(field_state: FieldState) -> Dict[str, float]:
    pos = field_state.position_12d
    logos_temps = decompose_temperature_by_logos(pos)
    strong_indicator = logos_temps.get("tangible", 0.0)
    weak_indicator = logos_temps.get("existential", 0.0)
    color_indicator = logos_temps.get("symbolic", 0.0)
    binding = strong_indicator * math.exp(-field_state.temperature / 10.0)
    balance = min(strong_indicator, weak_indicator, color_indicator)
    denom = max(strong_indicator, weak_indicator, color_indicator, 1e-10)
    stability = balance / denom
    decay_rate = math.exp(-stability * 10.0)
    return {
        "strong_force": strong_indicator,
        "weak_force": weak_indicator,
        "color_charge": color_indicator,
        "binding_energy": binding,
        "stability_score": stability,
        "decay_rate": decay_rate,
    }


def create_particle_from_state(
    field_state: FieldState,
    name: str = "unknown",
) -> ParticleLike:
    em_props = map_paraclete_to_em(field_state)
    grav_props = map_paraclete_to_gravity(field_state)
    nuclear_props = map_paraclete_to_nuclear(field_state)
    mass_like = grav_props["mass_indicator"]
    charge_like = em_props["electric_field"] / (em_props["magnetic_field"] + 1e-10)
    stability = nuclear_props["stability_score"]
    spin_like = em_props["field_ratio"] % 2.0
    return ParticleLike(
        name=name,
        state=field_state,
        stability_score=stability,
        mass_like=mass_like,
        charge_like=charge_like,
        spin_like=spin_like,
    )


def build_creation_hierarchy() -> CreationLayer:
    pregeom = CreationLayer(
        name="pre-geometric",
        descriptor={"description": "Fundamental field configurations"},
        scale=1e-35,
    )
    fields = CreationLayer(
        name="field-configuration",
        descriptor={"description": "Classical and quantum fields"},
        scale=1e-15,
    )
    particles = CreationLayer(
        name="particle-spectrum",
        descriptor={"description": "Elementary particles"},
        scale=1e-15,
    )
    atomic = CreationLayer(
        name="atomic",
        descriptor={"description": "Atoms and nuclei"},
        scale=1e-10,
    )
    molecular = CreationLayer(
        name="molecular",
        descriptor={"description": "Molecules and compounds"},
        scale=1e-9,
    )
    stellar = CreationLayer(
        name="stellar",
        descriptor={"description": "Stars and planets"},
        scale=1e9,
    )
    galactic = CreationLayer(
        name="galactic",
        descriptor={"description": "Galaxies and clusters"},
        scale=1e21,
    )
    symbolic = CreationLayer(
        name="symbolic-cognitive",
        descriptor={"description": "Information and consciousness"},
        scale=0.0,
    )
    root = CreationLayer(
        name="creation",
        descriptor={"description": "Complete creation hierarchy"},
        scale=1e30,
    )
    root.add_element(pregeom)
    root.add_element(fields)
    root.add_element(particles)
    root.add_element(atomic)
    root.add_element(molecular)
    root.add_element(stellar)
    root.add_element(galactic)
    root.add_element(symbolic)
    return root


def cluster_states_to_layers(
    states: List[FieldState],
    threshold: float = 1.0,
) -> List[CreationLayer]:
    if not states:
        return []

    layers: List[CreationLayer] = []
    clusters: Dict[str, List[FieldState]] = {
        "cold": [],
        "cool": [],
        "warm": [],
        "hot": [],
        "extreme": [],
    }

    for state in states:
        regime = classify_temperature_regime(state.temperature)
        if regime not in clusters:
            continue
        clusters[regime].append(state)

    scale_map = {
        "cold": 1e-15,
        "cool": 1e-10,
        "warm": 1e-5,
        "hot": 1.0,
        "extreme": 1e5,
    }

    for regime, cluster_states in clusters.items():
        if not cluster_states:
            continue
        layer = CreationLayer(
            name=f"{regime}-regime",
            descriptor={
                "temperature_regime": regime,
                "count": len(cluster_states),
                "threshold": threshold,
            },
            scale=scale_map.get(regime, 1.0),
        )
        for i, state in enumerate(cluster_states):
            particle = create_particle_from_state(state, f"{regime}_{i}")
            layer.add_element(particle)
        layers.append(layer)

    return layers


class PhysicsInterpreter:
    def __init__(self) -> None:
        self.hierarchy = build_creation_hierarchy()

    def interpret_state(self, field_state: FieldState) -> Dict[str, Any]:
        return {
            "electromagnetic": map_paraclete_to_em(field_state),
            "gravitational": map_paraclete_to_gravity(field_state),
            "nuclear": map_paraclete_to_nuclear(field_state),
            "temperature": field_state.temperature,
            "regime": classify_temperature_regime(field_state.temperature),
            "logos_decomposition": decompose_temperature_by_logos(
                field_state.position_12d
            ),
        }

    def classify_particle(self, particle: ParticleLike) -> str:
        if particle.mass_like < 0.001:
            if abs(particle.charge_like) < 0.1:
                return "neutrino-like"
            return "photon-like"
        if particle.mass_like < 1.0:
            if abs(particle.charge_like) > 0.5:
                return "lepton-like"
            return "meson-like"
        if particle.stability_score > 0.8:
            return "baryon-like"
        return "exotic"
