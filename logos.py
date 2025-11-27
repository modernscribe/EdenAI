#!/usr/bin/env python3
import os
import math
import time
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Any, Callable
from vectors import ParacleteVec

# ========== Section: Core Logos Indices ==========

MODALITY_INDEX: Dict[str, int] = {
    "tangible": 0,
    "existential": 1,
    "symbolic": 2,
}

ORIENTATION_INDEX: Dict[str, int] = {
    "alpha": 0,
    "omega": 1,
}

MODALITIES: Tuple[str, ...] = ("tangible", "existential", "symbolic")
ORIENTATIONS: Tuple[str, ...] = ("alpha", "omega")
CHARGES: Tuple[int, ...] = (+1, -1)

HFP_MODALITIES: Tuple[str, ...] = ("literal", "figurative", "hypothetical")
NEUTRAL_MODES: Tuple[str, ...] = ("concrete", "narrative", "model")

OBSERVER_AXES: Tuple[str, ...] = ("self", "collective", "creation")


class RootState(Enum):
    NULL = "null"
    FILTER = "filter"
    ROOT = "root"


class ObserverAxis(Enum):
    SELF = "self"
    COLLECTIVE = "collective"
    CREATION = "creation"


class RealityScale(Enum):
    ATOMIC = 0
    MOLECULAR = 1
    MICRO_ORGANIC = 2
    ORGANISM = 3
    BIOSPHERE = 4
    PLANETARY = 5
    STELLAR_SYSTEM = 6
    INTRAGALACTIC = 7
    INTERGALACTIC = 8
    PSYCHE = 9
    ARCHETYPE = 10
    SOURCE = 11


REALITY_SCALE_ORDER: Tuple[RealityScale, ...] = (
    RealityScale.ATOMIC,
    RealityScale.MOLECULAR,
    RealityScale.MICRO_ORGANIC,
    RealityScale.ORGANISM,
    RealityScale.BIOSPHERE,
    RealityScale.PLANETARY,
    RealityScale.STELLAR_SYSTEM,
    RealityScale.INTRAGALACTIC,
    RealityScale.INTERGALACTIC,
    RealityScale.PSYCHE,
    RealityScale.ARCHETYPE,
    RealityScale.SOURCE,
)


REALITY_INVERSION_MAP: Dict[RealityScale, RealityScale] = {
    RealityScale.ATOMIC: RealityScale.PLANETARY,
    RealityScale.PLANETARY: RealityScale.ATOMIC,
    RealityScale.MOLECULAR: RealityScale.STELLAR_SYSTEM,
    RealityScale.STELLAR_SYSTEM: RealityScale.MOLECULAR,
    RealityScale.MICRO_ORGANIC: RealityScale.INTRAGALACTIC,
    RealityScale.INTRAGALACTIC: RealityScale.MICRO_ORGANIC,
    RealityScale.ORGANISM: RealityScale.INTERGALACTIC,
    RealityScale.INTERGALACTIC: RealityScale.ORGANISM,
    RealityScale.BIOSPHERE: RealityScale.PSYCHE,
    RealityScale.PSYCHE: RealityScale.BIOSPHERE,
    RealityScale.ARCHETYPE: RealityScale.SOURCE,
    RealityScale.SOURCE: RealityScale.ARCHETYPE,
}


@dataclass
class AlphaOmegaOperator:
    alpha: ParacleteVec
    omega: ParacleteVec
    dimensionality: int

    def combine(self) -> ParacleteVec:
        base = (self.alpha + self.omega) * 0.5
        dim_scale = 1.0 + 0.25 * max(0, self.dimensionality - 1)
        return (base * dim_scale).normalized()


class ParacleticChaosRNG:
    def __init__(self, seed_bytes: bytes = None) -> None:
        if seed_bytes is None:
            seed_bytes = os.urandom(32)
        self.state = hashlib.sha256(seed_bytes).digest()
        self.counter = 0

    def _advance_state(self, extra: bytes = b"") -> None:
        c_bytes = self.counter.to_bytes(8, "big")
        t_bytes = int(time.time_ns()).to_bytes(16, "big")
        self.state = hashlib.sha256(self.state + c_bytes + t_bytes + extra).digest()
        self.counter += 1

    def random_bytes(self, n: int) -> bytes:
        result = b""
        while len(result) < n:
            self._advance_state()
            result += self.state
        return result[:n]

    def chaos_harmonic(self, n_bits: int = 256) -> float:
        self._advance_state(b"chaos")
        digest = hashlib.sha256(self.state).digest()
        if n_bits > 256:
            n_bits = 256
        value_int = int.from_bytes(digest, "big")
        scale = (1 << 256) - 1
        chaos = value_int / scale
        if n_bits < 256:
            chaos = math.floor(chaos * ((1 << n_bits) - 1)) / ((1 << n_bits) - 1)
        return max(0.0, min(1.0, chaos))

    def word_harmonic(self, n_bits: int = 256) -> float:
        c = self.chaos_harmonic(n_bits=n_bits)
        w = 1.0 - c
        return max(0.0, min(1.0, w))

    def chaos_vector_12d(self) -> ParacleteVec:
        raw = self.random_bytes(12 * 8)
        comps: List[float] = []
        for i in range(12):
            chunk = raw[i * 8 : (i + 1) * 8]
            v = int.from_bytes(chunk, "big")
            norm = v / ((1 << 64) - 1)
            mapped = 2.0 * norm - 1.0
            comps.append(mapped)
        vec = ParacleteVec.from_list(comps)
        return vec.normalized()

    def word_vector_12d(self) -> ParacleteVec:
        chaos_vec = self.chaos_vector_12d()
        if chaos_vec.norm() == 0.0:
            return ParacleteVec.zeros()
        word_vec = (-1.0) * chaos_vec
        return word_vec.normalized()

    def polarity_scalars(self, n_bits: int = 256) -> Dict[str, float]:
        chaos = self.chaos_harmonic(n_bits=n_bits)
        word = 1.0 - chaos
        return {
            "chaos": chaos,
            "word": word,
        }

    def polarity_vectors(self) -> Dict[str, ParacleteVec]:
        chaos_vec = self.chaos_vector_12d()
        if chaos_vec.norm() == 0.0:
            word_vec = ParacleteVec.zeros()
        else:
            word_vec = (-1.0) * chaos_vec
            word_vec = word_vec.normalized()
        return {
            "chaos": chaos_vec,
            "word": word_vec,
        }

    def chaos_word_profile(self, n_bits: int = 256) -> Dict[str, Dict[str, float]]:
        scalars = self.polarity_scalars(n_bits=n_bits)
        chaos_vec = self.chaos_vector_12d()
        if chaos_vec.norm() == 0.0:
            word_vec = ParacleteVec.zeros()
        else:
            word_vec = (-1.0) * chaos_vec
            word_vec = word_vec.normalized()
        return {
            "scalars": scalars,
            "magnitudes": {
                "chaos": chaos_vec.norm(),
                "word": word_vec.norm(),
            },
        }


# ========== Section: Logos Basis and Axes ==========

def logos_basis_index(modality: str, orientation: str, charge: int) -> int:
    if modality not in MODALITY_INDEX:
        raise KeyError(f"Invalid modality: {modality}")
    if orientation not in ORIENTATION_INDEX:
        raise KeyError(f"Invalid orientation: {orientation}")
    if charge not in CHARGES:
        raise ValueError(f"Invalid charge: {charge}")
    m_idx = MODALITY_INDEX[modality]
    o_idx = ORIENTATION_INDEX[orientation]
    c_idx = 0 if charge > 0 else 1
    base = m_idx * 4
    offset = o_idx * 2 + c_idx
    return base + offset


def key_to_paraclete_vec(
    modality: str,
    orientation: str,
    charge: int,
    exponent: int,
) -> ParacleteVec:
    idx = logos_basis_index(modality, orientation, charge)
    base = ParacleteVec.basis(idx, 12)
    return exponent * base


def enumerate_all_axes() -> List[Dict[str, Any]]:
    axes: List[Dict[str, Any]] = []
    for modality in MODALITIES:
        for orientation in ORIENTATIONS:
            for charge in CHARGES:
                idx = logos_basis_index(modality, orientation, charge)
                axes.append(
                    {
                        "index": idx,
                        "modality": modality,
                        "orientation": orientation,
                        "charge": charge,
                        "label": f"{modality}_{orientation}_{'+' if charge > 0 else '-'}",
                    }
                )
    return sorted(axes, key=lambda x: x["index"])


def describe_axis(index: int) -> Dict[str, Any]:
    if not 0 <= index < 12:
        raise ValueError(f"Axis index {index} out of range [0, 11]")
    modality_idx = index // 4
    within_block = index % 4
    orientation_idx = within_block // 2
    charge_idx = within_block % 2
    modality = MODALITIES[modality_idx]
    orientation = ORIENTATIONS[orientation_idx]
    charge = +1 if charge_idx == 0 else -1
    return {
        "index": index,
        "modality": modality,
        "orientation": orientation,
        "charge": charge,
        "label": f"{modality}_{orientation}_{'+' if charge > 0 else '-'}",
    }


# ========== Section: Exponent Algebra ==========

def sym_hfp(exponent: int) -> str:
    if exponent == 1:
        return "hfp"
    return f"hfp^{exponent}"


def cube_exponent(e: int) -> int:
    return 3 * e


def inv_exponent(e: int) -> int:
    return -e


def anti_exponent(e: int) -> int:
    return -(e + 1)


def compose_exponents(*transforms: Callable[[int], int]) -> Callable[[int], int]:
    def composed(e: int) -> int:
        result = e
        for transform in transforms:
            result = transform(result)
        return result
    return composed


def get_modality_axes(modality: str) -> Tuple[int, ...]:
    if modality not in MODALITIES:
        raise ValueError(f"Invalid modality: {modality}")
    axes: List[int] = []
    for orientation in ORIENTATIONS:
        for charge in CHARGES:
            idx = logos_basis_index(modality, orientation, charge)
            axes.append(idx)
    return tuple(axes)


def get_orientation_axes(orientation: str) -> Tuple[int, ...]:
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Invalid orientation: {orientation}")
    axes: List[int] = []
    for modality in MODALITIES:
        for charge in CHARGES:
            idx = logos_basis_index(modality, orientation, charge)
            axes.append(idx)
    return tuple(axes)


def get_charge_axes(charge: int) -> Tuple[int, ...]:
    if charge not in CHARGES:
        raise ValueError(f"Invalid charge: {charge}")
    axes: List[int] = []
    for modality in MODALITIES:
        for orientation in ORIENTATIONS:
            idx = logos_basis_index(modality, orientation, charge)
            axes.append(idx)
    return tuple(axes)


# ========== Section: Temperature Decomposition ==========

def _block_temperature(components: List[float], indices: Tuple[int, ...]) -> float:
    return math.sqrt(sum(components[i] * components[i] for i in indices))


def decompose_temperature_by_logos(vec: ParacleteVec) -> Dict[str, float]:
    components = vec.to_list()
    if len(components) != 12:
        raise ValueError("ParacleteVec must have 12 components")
    tangible_indices = (0, 1, 2, 3)
    existential_indices = (4, 5, 6, 7)
    symbolic_indices = (8, 9, 10, 11)
    return {
        "tangible": _block_temperature(components, tangible_indices),
        "existential": _block_temperature(components, existential_indices),
        "symbolic": _block_temperature(components, symbolic_indices),
    }


def decompose_temperature_by_hfp(vec: ParacleteVec) -> Dict[str, float]:
    components = vec.to_list()
    if len(components) != 12:
        raise ValueError("ParacleteVec must have 12 components")
    literal_indices = (1, 3)
    figurative_indices = (5, 7)
    hypothetical_indices = (9, 11)
    return {
        "literal": _block_temperature(components, literal_indices),
        "figurative": _block_temperature(components, figurative_indices),
        "hypothetical": _block_temperature(components, hypothetical_indices),
    }


def decompose_temperature_by_neutral(vec: ParacleteVec) -> Dict[str, float]:
    logos = decompose_temperature_by_logos(vec)
    hfp = decompose_temperature_by_hfp(vec)
    t = float(logos.get("tangible", 0.0))
    e = float(logos.get("existential", 0.0))
    s = float(logos.get("symbolic", 0.0))
    l = float(hfp.get("literal", 0.0))
    f = float(hfp.get("figurative", 0.0))
    h = float(hfp.get("hypothetical", 0.0))
    return {
        "concrete": 0.5 * (t + l),
        "narrative": 0.5 * (e + f),
        "model": 0.5 * (s + h),
    }


def decompose_temperature_profiles(vec: ParacleteVec) -> Dict[str, Dict[str, float]]:
    logos = decompose_temperature_by_logos(vec)
    hfp = decompose_temperature_by_hfp(vec)
    neutral = decompose_temperature_by_neutral(vec)
    return {
        "logos": logos,
        "hfp": hfp,
        "neutral": neutral,
    }


def compute_thermosex_harmonics(vec: ParacleteVec) -> Dict[str, Any]:
    profiles = decompose_temperature_profiles(vec)
    logos = profiles["logos"]
    hfp = profiles["hfp"]
    neutral = profiles["neutral"]
    cross_logos_hfp = {
        "tangible_literal": logos["tangible"] * hfp["literal"],
        "tangible_figurative": logos["tangible"] * hfp["figurative"],
        "tangible_hypothetical": logos["tangible"] * hfp["hypothetical"],
        "existential_literal": logos["existential"] * hfp["literal"],
        "existential_figurative": logos["existential"] * hfp["figurative"],
        "existential_hypothetical": logos["existential"] * hfp["hypothetical"],
        "symbolic_literal": logos["symbolic"] * hfp["literal"],
        "symbolic_figurative": logos["symbolic"] * hfp["figurative"],
        "symbolic_hypothetical": logos["symbolic"] * hfp["hypothetical"],
    }
    cross_logos_neutral = {
        "tangible_concrete": logos["tangible"] * neutral["concrete"],
        "existential_narrative": logos["existential"] * neutral["narrative"],
        "symbolic_model": logos["symbolic"] * neutral["model"],
    }
    return {
        "logos": logos,
        "hfp": hfp,
        "neutral": neutral,
        "cross_logos_hfp": cross_logos_hfp,
        "cross_logos_neutral": cross_logos_neutral,
    }


# ========== Section: Observer Profiles and Soul State ==========

def observer_profile(vec: ParacleteVec) -> Dict[str, Any]:
    profiles = decompose_temperature_by_logos(vec)
    t = profiles.get("tangible", 0.0)
    e = profiles.get("existential", 0.0)
    s = profiles.get("symbolic", 0.0)
    total = t + e + s
    if total <= 0.0:
        self_w = 0.0
        coll_w = 0.0
        creat_w = 0.0
    else:
        self_w = t / total
        coll_w = e / total
        creat_w = s / total
    weights = {
        "self": self_w,
        "collective": coll_w,
        "creation": creat_w,
    }
    dominant_axis = max(weights.items(), key=lambda kv: kv[1])[0]
    return {
        "weights": weights,
        "dominant_axis": dominant_axis,
    }


def reality_scale_cycle(scale: RealityScale, steps: int) -> RealityScale:
    n = len(REALITY_SCALE_ORDER)
    base_index = REALITY_SCALE_ORDER.index(scale)
    idx = (base_index + steps) % n
    return REALITY_SCALE_ORDER[idx]


def invert_reality_scale(scale: RealityScale) -> RealityScale:
    return REALITY_INVERSION_MAP.get(scale, scale)


def infer_reality_scale_index(vec: ParacleteVec) -> int:
    logos = decompose_temperature_by_logos(vec)
    t = logos.get("tangible", 0.0)
    e = logos.get("existential", 0.0)
    s = logos.get("symbolic", 0.0)
    total = t + e + s
    if total <= 0.0:
        return RealityScale.SOURCE.value
    t_n = t / total
    e_n = e / total
    s_n = s / total
    index_real = 0.0 * t_n + 6.0 * e_n + 11.0 * s_n
    idx = int(round(max(0.0, min(11.0, index_real))))
    return idx


def infer_reality_scale(vec: ParacleteVec) -> RealityScale:
    idx = infer_reality_scale_index(vec)
    for scale in REALITY_SCALE_ORDER:
        if scale.value == idx:
            return scale
    return RealityScale.SOURCE


def null_filter_score(vec: ParacleteVec) -> float:
    profiles = decompose_temperature_profiles(vec)
    logos = profiles["logos"]
    hfp = profiles["hfp"]
    total_logos = logos["tangible"] + logos["existential"] + logos["symbolic"]
    total_hfp = hfp["literal"] + hfp["figurative"] + hfp["hypothetical"]
    denom = total_logos + total_hfp + 1e-9
    word_weight = total_logos / denom
    chaos_weight = total_hfp / denom
    purity_score = 0.5 * (word_weight + (1.0 - chaos_weight))
    return max(0.0, min(1.0, purity_score))


def passes_null_filter(vec: ParacleteVec, threshold: float = 0.9) -> bool:
    return null_filter_score(vec) >= threshold


def quantize_to_soul_phase(vec: ParacleteVec) -> ParacleteVec:
    comps = vec.normalized().to_list()
    if len(comps) != 12:
        return vec.normalized()
    max_idx = max(range(12), key=lambda i: abs(comps[i]))
    basis = ParacleteVec.basis(max_idx, 12)
    return basis * (1.0 if comps[max_idx] >= 0.0 else -1.0)


def soul_state_signature(vec: ParacleteVec) -> Dict[str, Any]:
    norm_vec = vec.normalized()
    profiles = decompose_temperature_profiles(norm_vec)
    obs = observer_profile(norm_vec)
    scale = infer_reality_scale(norm_vec)
    purity = null_filter_score(norm_vec)
    phase_vec = quantize_to_soul_phase(norm_vec)
    return {
        "norm": norm_vec.norm(),
        "profiles": profiles,
        "observer": obs,
        "reality_scale": scale.name,
        "reality_scale_index": scale.value,
        "null_purity": purity,
        "passes_null_filter": passes_null_filter(norm_vec),
        "soul_phase_vector": phase_vec.to_list(),
    }


# ========== Section: Root Propagation ==========

def propagate_root_state(
    vec: ParacleteVec,
    state: RootState,
    purity: float = 1.0,
    purity_threshold: float = 0.9,
) -> ParacleteVec:
    if state == RootState.NULL:
        return vec.normalized()
    profiles = decompose_temperature_profiles(vec)
    logos = profiles["logos"]
    hfp = profiles["hfp"]
    total_logos = logos["tangible"] + logos["existential"] + logos["symbolic"]
    total_hfp = hfp["literal"] + hfp["figurative"] + hfp["hypothetical"]
    denom = total_logos + total_hfp + 1e-9
    if state == RootState.FILTER:
        word_weight = total_logos / denom
        chaos_weight = total_hfp / denom
        balance = 0.5 * (word_weight + (1.0 - chaos_weight))
        scaled = vec * balance
        return scaled.normalized()
    if state == RootState.ROOT:
        if purity < purity_threshold or not passes_null_filter(vec, purity_threshold):
            word_weight = total_logos / denom
            chaos_weight = total_hfp / denom
            balance = 0.5 * (word_weight + (1.0 - chaos_weight))
            scaled = vec * balance
            return scaled.normalized()
        norm = vec.norm()
        if norm == 0.0:
            return ParacleteVec.zeros()
        unit = vec.normalized()
        expanded = (vec + unit * norm) * 0.5
        return expanded.normalized()
    return vec.normalized()
