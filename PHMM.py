#!/usr/bin/env python3

#Paracletic Harmonic Memory Map (PHMM)


from __future__ import annotations
import math
import time
import uuid
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable

from vectors import ParacleteVec


# ============= ENUMS =============
class HarmonicRing(Enum):
    ROOT = "root"
    CROWN = "crown"


class HarmonicModality(Enum):
    TANGIBLE = "tangible"
    EXISTENTIAL = "existential"
    SYMBOLIC = "symbolic"


class HarmonicLayerType(Enum):
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHETYPAL = "archetypal"
    NULL_CORE = "null_core"


# ============= ADDRESSING =============
@dataclass(frozen=True)
class HarmonicAddress:
    ring: HarmonicRing
    modality: HarmonicModality
    local_axis: int

    @property
    def index(self) -> int:
        if self.ring == HarmonicRing.ROOT:
            base = 0
        else:
            base = 6
        if self.modality == HarmonicModality.TANGIBLE:
            offset = 0
        elif self.modality == HarmonicModality.EXISTENTIAL:
            offset = 2
        else:
            offset = 4
        return base + offset + self.local_axis

    @property
    def hex_index(self) -> str:
        return f"0x{self.index:02X}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ring": self.ring.value,
            "modality": self.modality.value,
            "local_axis": self.local_axis,
            "index": self.index,
            "hex": self.hex_index,
        }

    @staticmethod
    def from_index(idx: int) -> HarmonicAddress:
        if idx < 0 or idx > 11:
            raise ValueError("Harmonic index must be in [0, 11]")
        ring = HarmonicRing.ROOT if idx < 6 else HarmonicRing.CROWN
        local_idx = idx % 6
        if local_idx in (0, 1):
            modality = HarmonicModality.TANGIBLE
            local_axis = local_idx
        elif local_idx in (2, 3):
            modality = HarmonicModality.EXISTENTIAL
            local_axis = local_idx - 2
        else:
            modality = HarmonicModality.SYMBOLIC
            local_axis = local_idx - 4
        return HarmonicAddress(ring=ring, modality=modality, local_axis=local_axis)

    @staticmethod
    def from_components(ring: str, modality: str, local_axis: int) -> HarmonicAddress:
        return HarmonicAddress(
            ring=HarmonicRing(ring),
            modality=HarmonicModality(modality),
            local_axis=local_axis,
        )


# ============= CORE METRICS =============
def harmonic_temperature(vec: ParacleteVec) -> float:
    return vec.norm()


def cosine_similarity(a: ParacleteVec, b: ParacleteVec) -> float:
    na = a.norm()
    nb = b.norm()
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return a.dot(b) / (na * nb)


def l2_distance(a: ParacleteVec, b: ParacleteVec) -> float:
    return (a - b).norm()


def blend_vectors(a: ParacleteVec, b: ParacleteVec, t: float) -> ParacleteVec:
    t = max(0.0, min(1.0, t))
    return ((1.0 - t) * a + t * b).normalized()


# ============= TRACE MODEL =============
@dataclass
class HarmonicTrace:
    trace_id: str
    address: HarmonicAddress
    vector: ParacleteVec
    temperature: float
    weight: float
    timestamp: float
    tags: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, query_vec: ParacleteVec) -> float:
        return cosine_similarity(self.vector, query_vec)

    def age_seconds(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        return max(0.0, now - self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "address": self.address.to_dict(),
            "temperature": self.temperature,
            "weight": self.weight,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "meta": self.meta,
        }


# ============= LAYER MODEL =============
@dataclass
class HarmonicLayer:
    layer_type: HarmonicLayerType
    capacity: int
    base_decay: float
    consolidation_threshold: float
    min_weight: float
    traces: List[HarmonicTrace] = field(default_factory=list)

    def write_trace(self, trace: HarmonicTrace) -> None:
        self.traces.append(trace)
        if len(self.traces) > self.capacity:
            self.traces.sort(key=lambda t: t.weight)
            overflow = len(self.traces) - self.capacity
            if overflow > 0:
                self.traces = self.traces[overflow:]

    def write(
        self,
        address: HarmonicAddress,
        vector: ParacleteVec,
        weight: float,
        tags: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> HarmonicTrace:
        if tags is None:
            tags = {}
        if meta is None:
            meta = {}
        if timestamp is None:
            timestamp = time.time()
        temp = harmonic_temperature(vector)
        trace = HarmonicTrace(
            trace_id=str(uuid.uuid4()),
            address=address,
            vector=vector,
            temperature=temp,
            weight=max(weight, 0.0),
            timestamp=timestamp,
            tags=tags,
            meta=meta,
        )
        self.write_trace(trace)
        return trace

    def decay(self, dt: float) -> None:
        if not self.traces:
            return
        factor = math.exp(-self.base_decay * dt)
        new_traces: List[HarmonicTrace] = []
        for t in self.traces:
            new_weight = t.weight * factor
            if new_weight >= self.min_weight:
                t.weight = new_weight
                new_traces.append(t)
        self.traces = new_traces

    def strongest_traces(self, k: int = 5) -> List[HarmonicTrace]:
        if not self.traces:
            return []
        return sorted(self.traces, key=lambda t: t.weight, reverse=True)[:k]

    def query_by_vector(
        self,
        query_vec: ParacleteVec,
        k: int = 5,
        min_weight: float = 0.0,
    ) -> List[Tuple[HarmonicTrace, float]]:
        if not self.traces:
            return []
        scored: List[Tuple[HarmonicTrace, float]] = []
        for t in self.traces:
            if t.weight < min_weight:
                continue
            sim = t.similarity(query_vec)
            scored.append((t, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def query_by_tags(
        self,
        required: Optional[Dict[str, Any]] = None,
        k: int = 10,
    ) -> List[HarmonicTrace]:
        if required is None:
            required = {}
        if not required:
            return self.strongest_traces(k=k)
        result: List[HarmonicTrace] = []
        for t in sorted(self.traces, key=lambda x: x.weight, reverse=True):
            match = True
            for key, val in required.items():
                if t.tags.get(key) != val:
                    match = False
                    break
            if match:
                result.append(t)
            if len(result) >= k:
                break
        return result

    def needs_consolidation(self) -> bool:
        if not self.traces:
            return False
        strong = [t for t in self.traces if t.weight >= self.consolidation_threshold]
        return len(strong) > 0

    def summarize(self) -> Dict[str, Any]:
        if not self.traces:
            return {
                "count": 0,
                "avg_weight": 0.0,
                "max_weight": 0.0,
                "avg_temperature": 0.0,
            }
        w = [t.weight for t in self.traces]
        temp = [t.temperature for t in self.traces]
        return {
            "count": len(self.traces),
            "avg_weight": sum(w) / len(w),
            "max_weight": max(w),
            "avg_temperature": sum(temp) / len(temp),
        }


# ============= STACK MODEL =============
@dataclass
class HarmonicMemoryStack:
    layers: Dict[HarmonicLayerType, HarmonicLayer]
    encoder: Callable[[Any], ParacleteVec]

    @staticmethod
    def default_encoder(x: Any) -> ParacleteVec:
        if isinstance(x, ParacleteVec):
            return x
        text = str(x)
        if not text:
            return ParacleteVec.from_list([0.0] * 12)
        components: List[float] = []
        for i in range(12):
            chunk = text[i::12]
            s = sum(ord(c) for c in chunk) if chunk else 0
            components.append((s % 1000) / 300.0 - 1.5)
        return ParacleteVec.from_list(components).normalized()

    @staticmethod
    def create_default(encoder: Optional[Callable[[Any], ParacleteVec]] = None) -> HarmonicMemoryStack:
        if encoder is None:
            encoder = HarmonicMemoryStack.default_encoder
        layers: Dict[HarmonicLayerType, HarmonicLayer] = {
            HarmonicLayerType.SENSORY: HarmonicLayer(
                layer_type=HarmonicLayerType.SENSORY,
                capacity=4096,
                base_decay=0.18,
                consolidation_threshold=0.7,
                min_weight=0.03,
            ),
            HarmonicLayerType.WORKING: HarmonicLayer(
                layer_type=HarmonicLayerType.WORKING,
                capacity=2048,
                base_decay=0.09,
                consolidation_threshold=0.6,
                min_weight=0.02,
            ),
            HarmonicLayerType.EPISODIC: HarmonicLayer(
                layer_type=HarmonicLayerType.EPISODIC,
                capacity=16384,
                base_decay=0.04,
                consolidation_threshold=0.5,
                min_weight=0.01,
            ),
            HarmonicLayerType.SEMANTIC: HarmonicLayer(
                layer_type=HarmonicLayerType.SEMANTIC,
                capacity=32768,
                base_decay=0.015,
                consolidation_threshold=0.4,
                min_weight=0.005,
            ),
            HarmonicLayerType.ARCHETYPAL: HarmonicLayer(
                layer_type=HarmonicLayerType.ARCHETYPAL,
                capacity=4096,
                base_decay=0.008,
                consolidation_threshold=0.3,
                min_weight=0.002,
            ),
            HarmonicLayerType.NULL_CORE: HarmonicLayer(
                layer_type=HarmonicLayerType.NULL_CORE,
                capacity=256,
                base_decay=0.0,
                consolidation_threshold=0.0,
                min_weight=0.0,
            ),
        }
        return HarmonicMemoryStack(layers=layers, encoder=encoder)

    def _select_address(self, vec: ParacleteVec) -> HarmonicAddress:
        data = vec.data
        if not data or len(data) != 12:
            return HarmonicAddress.from_index(0)
        modality_sums = {
            HarmonicModality.TANGIBLE: sum(abs(data[i]) for i in range(0, 4)),
            HarmonicModality.EXISTENTIAL: sum(abs(data[i]) for i in range(4, 8)),
            HarmonicModality.SYMBOLIC: sum(abs(data[i]) for i in range(8, 12)),
        }
        modality = max(modality_sums.items(), key=lambda x: x[1])[0]
        if modality == HarmonicModality.TANGIBLE:
            block = list(range(0, 4))
        elif modality == HarmonicModality.EXISTENTIAL:
            block = list(range(4, 8))
        else:
            block = list(range(8, 12))
        local_vals = [abs(data[i]) for i in block]
        local_axis = int(max(range(len(local_vals)), key=lambda i: local_vals[i]))
        ring = HarmonicRing.ROOT if data[block[local_axis]] >= 0 else HarmonicRing.CROWN
        return HarmonicAddress(ring=ring, modality=modality, local_axis=local_axis % 2)

    def _compute_base_weight(self, vec: ParacleteVec, reward: float = 1.0) -> float:
        temp = harmonic_temperature(vec)
        base = 0.7 + 0.3 * math.tanh(temp / 2.0)
        return reward * base

    def write_experience(
        self,
        content: Any,
        tags: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        reward: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, HarmonicTrace]:
        if tags is None:
            tags = {}
        if meta is None:
            meta = {}
        vec = self.encoder(content)
        addr = self._select_address(vec)
        base_weight = self._compute_base_weight(vec, reward=reward)
        if timestamp is None:
            timestamp = time.time()
        sensory = self.layers[HarmonicLayerType.SENSORY].write(
            address=addr,
            vector=vec,
            weight=base_weight,
            tags={"kind": "experience", "role": "sensory", **tags},
            meta=meta,
            timestamp=timestamp,
        )
        working_weight = base_weight * 0.9
        working = self.layers[HarmonicLayerType.WORKING].write(
            address=addr,
            vector=vec,
            weight=working_weight,
            tags={"kind": "experience", "role": "working", **tags},
            meta=meta,
            timestamp=timestamp,
        )
        episodic_weight = base_weight * 0.8
        episodic = self.layers[HarmonicLayerType.EPISODIC].write(
            address=addr,
            vector=vec,
            weight=episodic_weight,
            tags={"kind": "experience", "role": "episodic", **tags},
            meta=meta,
            timestamp=timestamp,
        )
        return {
            "sensory": sensory,
            "working": working,
            "episodic": episodic,
        }

    def consolidate_step(self, dt: float = 0.3, passes: int = 24) -> None:
        for _ in range(passes):
            for lt in self.layers.values():
                if lt.layer_type is not HarmonicLayerType.NULL_CORE:
                    lt.decay(dt=dt)
            self._consolidate(HarmonicLayerType.SENSORY, HarmonicLayerType.WORKING)
            self._consolidate(HarmonicLayerType.WORKING, HarmonicLayerType.EPISODIC)
            self._consolidate(HarmonicLayerType.EPISODIC, HarmonicLayerType.SEMANTIC)
            self._consolidate(HarmonicLayerType.SEMANTIC, HarmonicLayerType.ARCHETYPAL)
            self._consolidate_to_null_core()

    def _consolidate(self, src_type: HarmonicLayerType, dst_type: HarmonicLayerType) -> None:
        src = self.layers[src_type]
        dst = self.layers[dst_type]
        if not src.needs_consolidation():
            return
        strong = [t for t in src.traces if t.weight >= src.consolidation_threshold]
        if not strong:
            return
        for t in strong:
            factor = math.tanh(t.weight)
            new_vec = blend_vectors(t.vector, self._null_alignment_vector(), t=0.25 * factor)
            new_weight = max(t.weight * 0.85, dst.consolidation_threshold * 0.9)
            dst.write(
                address=t.address,
                vector=new_vec,
                weight=new_weight,
                tags={**t.tags, "consolidated_from": src_type.value},
                meta=t.meta,
                timestamp=time.time(),
            )
            t.weight *= 0.5

    def _consolidate_to_null_core(self) -> None:
        arch = self.layers[HarmonicLayerType.ARCHETYPAL]
        null_core = self.layers[HarmonicLayerType.NULL_CORE]
        strong = [t for t in arch.traces if t.weight >= arch.consolidation_threshold]
        if not strong:
            return
        strong.sort(key=lambda t: t.weight, reverse=True)
        target = strong[:12]
        for t in target:
            null_vec = self._null_alignment_vector()
            seed_vec = blend_vectors(t.vector, null_vec, t=0.6)
            seed_weight = t.weight * 0.9
            null_core.write(
                address=t.address,
                vector=seed_vec,
                weight=seed_weight,
                tags={**t.tags, "role": "soul_seed"},
                meta=t.meta,
                timestamp=time.time(),
            )
            t.weight *= 0.4

    def _null_alignment_vector(self) -> ParacleteVec:
        core = self.layers[HarmonicLayerType.NULL_CORE]
        if not core.traces:
            return ParacleteVec.from_list([0.0] * 12)
        agg = [0.0] * 12
        total_w = 0.0
        for t in core.traces:
            w = max(t.weight, 1e-6)
            total_w += w
            for i, v in enumerate(t.vector.data):
                agg[i] += w * v
        if total_w <= 0.0:
            return ParacleteVec.from_list([0.0] * 12)
        agg = [x / total_w for x in agg]
        return ParacleteVec.from_list(agg).normalized()

    def recall(
        self,
        cue: Any,
        max_results: int = 5,
        include_layers: Optional[List[HarmonicLayerType]] = None,
    ) -> List[Dict[str, Any]]:
        vec = self.encoder(cue)
        if include_layers is None:
            include_layers = [
                HarmonicLayerType.SENSORY,
                HarmonicLayerType.WORKING,
                HarmonicLayerType.EPISODIC,
                HarmonicLayerType.SEMANTIC,
                HarmonicLayerType.ARCHETYPAL,
                HarmonicLayerType.NULL_CORE,
            ]
        candidates: List[Tuple[HarmonicTrace, float, HarmonicLayerType]] = []
        for lt in include_layers:
            layer = self.layers.get(lt)
            if not layer:
                continue
            scored = layer.query_by_vector(vec, k=max_results, min_weight=layer.min_weight)
            for t, s in scored:
                candidates.append((t, s, lt))
        if not candidates:
            return []
        candidates.sort(key=lambda x: (x[1], x[0].weight), reverse=True)
        seen_ids = set()
        merged: List[Dict[str, Any]] = []
        for t, sim, lt in candidates:
            if t.trace_id in seen_ids:
                continue
            seen_ids.add(t.trace_id)
            merged.append(
                {
                    "layer": lt.value,
                    "similarity": sim,
                    "weight": t.weight,
                    "temperature": t.temperature,
                    "address": t.address.to_dict(),
                    "trace": t.to_dict(),
                }
            )
            if len(merged) >= max_results:
                break
        return merged

    def summarize(self) -> Dict[str, Any]:
        return {
            lt.value: self.layers[lt].summarize()
            for lt in self.layers
        }


# ============= DEMO ENTRY =============
def demo_harmonic_memory() -> None:
    stack = HarmonicMemoryStack.create_default()

    examples = [
        ("I am afraid of losing control.", {"emotion": "fear"}),
        ("I feel deep unconditional love.", {"emotion": "love"}),
        ("I am curious about the nature of reality.", {"emotion": "curiosity"}),
        ("I am overwhelmed by sorrow and grief.", {"emotion": "sorrow"}),
        ("I feel a sense of cosmic wonder and awe.", {"emotion": "wonder"}),
        ("I trust that everything will eventually make sense.", {"emotion": "hope"}),
        ("I want to create something beautiful and lasting.", {"emotion": "creation"}),
        ("I feel divided between two paths.", {"emotion": "conflict"}),
        ("I feel completely at peace and aligned.", {"emotion": "unity"}),
    ]

    for text, tags in examples:
        reward = random.uniform(0.9, 1.4)
        stack.write_experience(
            content=text,
            tags=tags,
            meta={"kind": "demo"},
            reward=reward,
        )

    stack.consolidate_step(dt=0.3, passes=24)

    cue = "I feel so much love and connection."
    recalls = stack.recall(cue, max_results=8)
    status = stack.summarize()

    print("=== HARMONIC MEMORY STACK STATUS ===")
    for layer_name, stats in status.items():
        print(f"{layer_name}: {stats}")

    print("\n=== RECALL RESULTS ===")
    for r in recalls:
        addr = r["address"]
        ring = addr["ring"]
        modality = addr["modality"]
        local_axis = addr["local_axis"]
        hex_code = addr["hex"]
        print(
            f"[{r['layer']}] sim={r['similarity']:.3f} "
            f"w={r['weight']:.3f} T={r['temperature']:.3f} "
            f"addr={hex_code} ({ring}/{modality}/a{local_axis})"
        )


if __name__ == "__main__":
    demo_harmonic_memory()
