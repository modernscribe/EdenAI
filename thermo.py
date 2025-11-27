"""
Thermodynamic State and Evolution

This module implements thermodynamic concepts in the 12D Paraclete space,
including state variables, energy models, and relaxation dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum
import math
import random

from vectors import ParacleteVec
from phase_tree import PhaseTree, PhaseNode
from context import PhaseContext


class EnergyModel(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    LOGARITHMIC = "logarithmic"
    CUSTOM = "custom"


@dataclass
class ThermoState:
    node_id: str
    position: ParacleteVec
    temperature: float
    energy: float
    entropy: float
    heat_capacity: float = 1.0
    free_energy: float = 0.0
    pressure_like: float = 1.0
    volume_like: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "position": self.position.to_list(),
            "temperature": self.temperature,
            "energy": self.energy,
            "entropy": self.entropy,
            "heat_capacity": self.heat_capacity,
            "free_energy": self.free_energy,
            "pressure_like": self.pressure_like,
            "volume_like": self.volume_like,
            "metadata": self.metadata,
        }


@dataclass
class ThermoUniverse:
    tree: PhaseTree
    context: PhaseContext
    states: Dict[str, ThermoState]
    k_energy: float = 1.0
    k_entropy: float = 1.0
    k_relax: float = 0.1
    energy_model: EnergyModel = EnergyModel.LINEAR
    custom_energy_func: Optional[Callable[[float], float]] = None
    relax_model: str = "linear"
    time: float = 0.0
    history: List[Dict] = field(default_factory=list)

    @staticmethod
    def initialize(
        tree: PhaseTree,
        context: PhaseContext,
        energy_model: EnergyModel = EnergyModel.LINEAR,
        k_energy: float = 1.0,
        k_entropy: float = 1.0,
        k_relax: float = 0.1,
        custom_energy_func: Optional[Callable[[float], float]] = None,
        relax_model: str = "linear",
    ) -> ThermoUniverse:
        states: Dict[str, ThermoState] = {}
        ctx = context.combined

        for nid, node in tree.index.items():
            pos = node.to_vec() + ctx
            T = pos.norm()

            if energy_model == EnergyModel.CUSTOM and custom_energy_func is not None:
                E = custom_energy_func(T)
            elif energy_model == EnergyModel.LINEAR:
                E = k_energy * T
            elif energy_model == EnergyModel.QUADRATIC:
                E = k_energy * T * T
            elif energy_model == EnergyModel.LOGARITHMIC:
                E = k_energy * math.log1p(T)
            else:
                E = k_energy * T

            S = k_entropy * math.log1p(T) if T > 0 else 0.0
            F = E - T * S if T > 0 else E

            states[nid] = ThermoState(
                node_id=nid,
                position=pos,
                temperature=T,
                energy=E,
                entropy=S,
                free_energy=F,
                metadata={
                    "modality": node.key.modality,
                    "orientation": node.key.orientation,
                    "charge": node.key.charge,
                    "depth": node.key.depth,
                },
            )

        return ThermoUniverse(
            tree=tree,
            context=context,
            states=states,
            k_energy=k_energy,
            k_entropy=k_entropy,
            k_relax=k_relax,
            energy_model=energy_model,
            custom_energy_func=custom_energy_func,
            relax_model=relax_model,
        )

    def compute_energy(self, temperature: float) -> float:
        if self.custom_energy_func and self.energy_model == EnergyModel.CUSTOM:
            return self.custom_energy_func(temperature)

        if self.energy_model == EnergyModel.LINEAR:
            return self.k_energy * temperature
        if self.energy_model == EnergyModel.QUADRATIC:
            return self.k_energy * temperature * temperature
        if self.energy_model == EnergyModel.LOGARITHMIC:
            return self.k_energy * math.log1p(temperature)
        return self.k_energy * temperature

    def step(self, dt: float = 0.1, noise: float = 0.0) -> None:
        ctx = self.context.combined
        new_states: Dict[str, ThermoState] = {}

        for nid, st in self.states.items():
            node = self.tree.index[nid]
            target = node.to_vec() + ctx
            relax_dir = target - st.position

            alpha_sign = 1.0 if node.key.orientation == "alpha" else -1.0
            charge_sign = 1.0 if node.key.charge > 0 else -1.0
            base_rate = self.k_relax * alpha_sign * charge_sign * dt

            if noise > 0:
                noise_vec = ParacleteVec.from_list(
                    [random.gauss(0.0, noise) for _ in range(12)]
                )
                relax_dir = relax_dir + noise_vec

            distance = relax_dir.norm()

            if self.relax_model == "exponential":
                delta = RelaxationStrategy.exponential(relax_dir, base_rate, distance)
            elif self.relax_model == "sigmoid":
                delta = RelaxationStrategy.sigmoid(relax_dir, base_rate, distance)
            elif self.relax_model == "oscillatory":
                delta = RelaxationStrategy.oscillatory(relax_dir, base_rate, self.time)
            else:
                delta = RelaxationStrategy.linear(relax_dir, base_rate)

            new_pos = st.position + delta

            T = new_pos.norm()
            E = self.compute_energy(T)
            S = self.k_entropy * math.log1p(T) if T > 0 else 0.0
            F = E - T * S if T > 0 else E

            new_states[nid] = ThermoState(
                node_id=nid,
                position=new_pos,
                temperature=T,
                energy=E,
                entropy=S,
                heat_capacity=st.heat_capacity,
                free_energy=F,
                pressure_like=st.pressure_like,
                volume_like=st.volume_like,
                metadata=st.metadata,
            )

        self.states = new_states
        self.time += dt
        self.record_history()

    def record_history(self) -> None:
        self.history.append(
            {
                "time": self.time,
                "total_energy": self.total_energy(),
                "total_entropy": self.total_entropy(),
                "avg_temperature": self.average_temperature(),
            }
        )

    def total_energy(self) -> float:
        return sum(s.energy for s in self.states.values())

    def total_entropy(self) -> float:
        return sum(s.entropy for s in self.states.values())

    def average_temperature(self) -> float:
        temps = [s.temperature for s in self.states.values()]
        return sum(temps) / len(temps) if temps else 0.0

    def sample(self, n: int = 10) -> List[ThermoState]:
        items = list(self.states.values())
        random.shuffle(items)
        return items[: min(n, len(items))]

    def get_energy_spectrum(self) -> Dict[str, float]:
        spectrum = {"tangible": 0.0, "existential": 0.0, "symbolic": 0.0}
        for state in self.states.values():
            modality = state.metadata.get("modality", "tangible")
            spectrum[modality] += state.energy
        return spectrum

    def get_entropy_spectrum(self) -> Dict[str, float]:
        spectrum = {"tangible": 0.0, "existential": 0.0, "symbolic": 0.0}
        for state in self.states.values():
            modality = state.metadata.get("modality", "tangible")
            spectrum[modality] += state.entropy
        return spectrum

    def get_per_axis_energy(self) -> List[float]:
        axis_energy = [0.0] * 12
        for state in self.states.values():
            pos_data = state.position.data
            denom = state.position.norm_squared()
            if denom <= 0.0:
                continue
            scale = state.energy / denom
            for i in range(12):
                axis_energy[i] += scale * (pos_data[i] ** 2)
        return axis_energy

    def export_time_series(self) -> Tuple[List[float], Dict[str, List[float]]]:
        if not self.history:
            return [], {}
        times = [h["time"] for h in self.history]
        data = {
            "total_energy": [h["total_energy"] for h in self.history],
            "total_entropy": [h["total_entropy"] for h in self.history],
            "avg_temperature": [h["avg_temperature"] for h in self.history],
        }
        return times, data

    def equilibrate(self, steps: int = 100, dt: float = 0.1, tolerance: float = 1e-6) -> int:
        prev_energy = self.total_energy()
        for i in range(steps):
            self.step(dt)
            curr_energy = self.total_energy()
            if abs(curr_energy - prev_energy) < tolerance:
                return i + 1
            prev_energy = curr_energy
        return steps

    def get_state(self, node_id: str) -> Optional[ThermoState]:
        return self.states.get(node_id)

    def update_context(self, context: PhaseContext) -> None:
        self.context = context

    def reset_history(self) -> None:
        self.history = []
        self.time = 0.0


class RelaxationStrategy:
    @staticmethod
    def linear(relax_dir: ParacleteVec, rate: float) -> ParacleteVec:
        return rate * relax_dir

    @staticmethod
    def exponential(relax_dir: ParacleteVec, rate: float, distance: float) -> ParacleteVec:
        exp_rate = rate * math.exp(-distance)
        return exp_rate * relax_dir

    @staticmethod
    def sigmoid(relax_dir: ParacleteVec, rate: float, distance: float) -> ParacleteVec:
        sig_rate = rate * (2.0 / (1.0 + math.exp(-distance)) - 1.0)
        return sig_rate * relax_dir

    @staticmethod
    def oscillatory(
        relax_dir: ParacleteVec,
        rate: float,
        time: float,
        freq: float = 1.0,
    ) -> ParacleteVec:
        osc_rate = rate * (1.0 + 0.5 * math.sin(2.0 * math.pi * freq * time))
        return osc_rate * relax_dir
