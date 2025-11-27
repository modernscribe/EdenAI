"""
Temperature-Based Metrics and Decomposition

This module provides functions for computing and analyzing temperature
(as the norm of position vectors) in the 12D Paraclete space.
"""

from typing import Dict, List, Tuple, Optional
import math

from vectors import ParacleteVec
from logos import get_modality_axes, get_orientation_axes, get_charge_axes
from truth import ParacleteTruthRow


def compute_temperature(position: ParacleteVec) -> float:
    return position.norm()


def compute_energy_from_temperature(
    T: float,
    k_energy: float = 1.0,
    model: str = "linear",
) -> float:
    if model == "linear":
        return k_energy * T
    if model == "quadratic":
        return k_energy * T * T
    if model == "logarithmic":
        return k_energy * math.log1p(T) if T > 0.0 else 0.0
    raise ValueError(f"Unknown energy model: {model}")


def compute_entropy_from_temperature(
    T: float,
    k_entropy: float = 1.0,
    model: str = "logarithmic",
) -> float:
    if model == "logarithmic":
        return k_entropy * math.log1p(T) if T > 0.0 else 0.0
    if model == "linear":
        return k_entropy * T
    raise ValueError(f"Unknown entropy model: {model}")


def decompose_temperature_by_logos(position: ParacleteVec) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for modality in ["tangible", "existential", "symbolic"]:
        axes = get_modality_axes(modality)
        projected = position.project_onto_axes(tuple(axes))
        result[modality] = projected.norm()
    return result


def decompose_temperature_by_orientation(position: ParacleteVec) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for orientation in ["alpha", "omega"]:
        axes = get_orientation_axes(orientation)
        projected = position.project_onto_axes(tuple(axes))
        result[orientation] = projected.norm()
    return result


def decompose_temperature_by_charge(position: ParacleteVec) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for charge in (+1, -1):
        axes = get_charge_axes(charge)
        projected = position.project_onto_axes(tuple(axes))
        key = "positive" if charge > 0 else "negative"
        result[key] = projected.norm()
    return result


def compute_temperature_gradient(
    position: ParacleteVec,
    epsilon: float = 1e-6,
) -> ParacleteVec:
    norm = position.norm()
    if norm < epsilon:
        return ParacleteVec.zeros()
    return (1.0 / norm) * position


def compute_temperature_laplacian(position: ParacleteVec) -> float:
    norm = position.norm()
    if norm < 1e-10:
        return 0.0
    return 11.0 / norm


def classify_temperature_regime(T: float) -> str:
    if T < 0.1:
        return "cold"
    if T < 1.0:
        return "cool"
    if T < 10.0:
        return "warm"
    if T < 100.0:
        return "hot"
    return "extreme"


def lift_truth_row_to_temperature(
    paraclete_row: ParacleteTruthRow,
    k_energy: float = 1.0,
    k_entropy: float = 1.0,
) -> Dict:
    T_out = paraclete_row.temperature
    E_out = compute_energy_from_temperature(T_out, k_energy)
    S_out = compute_entropy_from_temperature(T_out, k_entropy)
    logos_decomp = decompose_temperature_by_logos(paraclete_row.output)
    orient_decomp = decompose_temperature_by_orientation(paraclete_row.output)
    charge_decomp = decompose_temperature_by_charge(paraclete_row.output)
    gradient = compute_temperature_gradient(paraclete_row.output)
    laplacian = compute_temperature_laplacian(paraclete_row.output)
    regime = classify_temperature_regime(T_out)
    return {
        "temperature": T_out,
        "energy": E_out,
        "entropy": S_out,
        "regime": regime,
        "logos_temperatures": logos_decomp,
        "orientation_temperatures": orient_decomp,
        "charge_temperatures": charge_decomp,
        "gradient_norm": gradient.norm(),
        "laplacian": laplacian,
        "position_3d": paraclete_row.position_3d,
        "axis_projections": list(paraclete_row.output.data),
        "boolean_inputs": paraclete_row.metadata.get("boolean_inputs", {}),
        "boolean_output": paraclete_row.metadata.get("boolean_output", False),
    }


def analyze_temperature_distribution(
    positions: List[ParacleteVec],
) -> Dict:
    if not positions:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
        }

    temps = [p.norm() for p in positions]
    temps.sort()
    n = len(temps)
    mean = sum(temps) / float(n)
    variance = sum((t - mean) ** 2 for t in temps) / float(n) if n > 0 else 0.0
    std = math.sqrt(variance)

    if n % 2 == 1:
        median = temps[n // 2]
    else:
        median = 0.5 * (temps[n // 2 - 1] + temps[n // 2])

    q1 = temps[n // 4] if n >= 4 else temps[0]
    q3 = temps[(3 * n) // 4] if n >= 4 else temps[-1]

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": temps[0],
        "max": temps[-1],
        "median": median,
        "q1": q1,
        "q3": q3,
    }


def compute_temperature_correlation(
    positions1: List[ParacleteVec],
    positions2: List[ParacleteVec],
) -> float:
    if len(positions1) != len(positions2):
        raise ValueError("Position lists must have same length")
    n = len(positions1)
    if n < 2:
        return 0.0

    temps1 = [p.norm() for p in positions1]
    temps2 = [p.norm() for p in positions2]

    mean1 = sum(temps1) / float(n)
    mean2 = sum(temps2) / float(n)

    cov = sum((t1 - mean1) * (t2 - mean2) for t1, t2 in zip(temps1, temps2))
    var1 = sum((t - mean1) ** 2 for t in temps1)
    var2 = sum((t - mean2) ** 2 for t in temps2)

    if var1 < 1e-10 or var2 < 1e-10:
        return 0.0

    return cov / math.sqrt(var1 * var2)


class TemperatureField:
    def __init__(self, positions: List[ParacleteVec]):
        self.positions = positions
        self.temperatures = [p.norm() for p in positions]

    def gradient_field(self) -> List[ParacleteVec]:
        return [compute_temperature_gradient(p) for p in self.positions]

    def laplacian_field(self) -> List[float]:
        return [compute_temperature_laplacian(p) for p in self.positions]

    def heat_flow(self, dt: float = 0.1) -> List[ParacleteVec]:
        new_positions: List[ParacleteVec] = []
        laplacians = self.laplacian_field()
        for pos, lap in zip(self.positions, laplacians):
            grad = compute_temperature_gradient(pos)
            delta = dt * lap * grad
            new_pos = pos + delta
            new_positions.append(new_pos)
        return new_positions
