"""
12-Dimensional Paraclete Vector Operations

This module provides pure mathematical operations on 12D vectors without
any Paraclete-specific semantics. All vectors are immutable and operations
return new vectors.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable, List
import math


# =========== Section: 12D Vector Type ===========

@dataclass(frozen=True)
class ParacleteVec:
    data: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.data) != 12:
            raise ValueError(f"ParacleteVec must be 12-dimensional, got {len(self.data)}")

    # ---------- Constructors ----------

    @staticmethod
    def zeros(dim: int = 12) -> ParacleteVec:
        if dim != 12:
            raise ValueError("ParacleteVec dimension must be 12")
        return ParacleteVec(tuple(0.0 for _ in range(dim)))

    @staticmethod
    def basis(i: int, dim: int = 12) -> ParacleteVec:
        if dim != 12:
            raise ValueError("ParacleteVec dimension must be 12")
        if not 0 <= i < dim:
            raise ValueError(f"Basis index {i} out of range [0, {dim-1}]")
        return ParacleteVec(tuple(1.0 if k == i else 0.0 for k in range(dim)))

    @staticmethod
    def from_list(data: List[float]) -> ParacleteVec:
        if len(data) != 12:
            raise ValueError(f"Expected 12 components, got {len(data)}")
        return ParacleteVec(tuple(float(x) for x in data))

    # ---------- Basic Protocol ----------

    def __len__(self) -> int:
        return 12

    def __iter__(self) -> Iterable[float]:
        return iter(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    # ---------- Arithmetic Ops ----------

    def __add__(self, other: ParacleteVec) -> ParacleteVec:
        if not isinstance(other, ParacleteVec):
            raise TypeError(f"Cannot add ParacleteVec and {type(other)}")
        return ParacleteVec(tuple(a + b for a, b in zip(self.data, other.data)))

    def __sub__(self, other: ParacleteVec) -> ParacleteVec:
        if not isinstance(other, ParacleteVec):
            raise TypeError(f"Cannot subtract {type(other)} from ParacleteVec")
        return ParacleteVec(tuple(a - b for a, b in zip(self.data, other.data)))

    def __mul__(self, scalar: float) -> ParacleteVec:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot multiply ParacleteVec by {type(scalar)}")
        return ParacleteVec(tuple(float(scalar) * a for a in self.data))

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> ParacleteVec:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot divide ParacleteVec by {type(scalar)}")
        if abs(scalar) < 1e-15:
            raise ZeroDivisionError("Division by zero in ParacleteVec.__truediv__")
        inv = 1.0 / float(scalar)
        return ParacleteVec(tuple(inv * a for a in self.data))

    def __neg__(self) -> ParacleteVec:
        return ParacleteVec(tuple(-a for a in self.data))

    # ---------- Core Linear Algebra ----------

    def dot(self, other: ParacleteVec) -> float:
        if not isinstance(other, ParacleteVec):
            raise TypeError(f"Cannot compute dot product with {type(other)}")
        return float(sum(a * b for a, b in zip(self.data, other.data)))

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def norm_squared(self) -> float:
        return self.dot(self)

    def normalized(self) -> ParacleteVec:
        n = self.norm()
        if n < 1e-10:
            return self
        inv = 1.0 / n
        return ParacleteVec(tuple(inv * a for a in self.data))

    def elementwise_mul(self, other: ParacleteVec) -> ParacleteVec:
        if not isinstance(other, ParacleteVec):
            raise TypeError(f"Cannot elementwise-multiply ParacleteVec and {type(other)}")
        return ParacleteVec(tuple(a * b for a, b in zip(self.data, other.data)))

    def clamp(self, min_value: float, max_value: float) -> ParacleteVec:
        lo = float(min_value)
        hi = float(max_value)
        if lo > hi:
            lo, hi = hi, lo
        return ParacleteVec(tuple(min(max(a, lo), hi) for a in self.data))

    def lerp(self, other: ParacleteVec, t: float) -> ParacleteVec:
        if not isinstance(other, ParacleteVec):
            raise TypeError(f"Cannot lerp ParacleteVec with {type(other)}")
        alpha = float(t)
        beta = 1.0 - alpha
        return ParacleteVec(tuple(beta * a + alpha * b for a, b in zip(self.data, other.data)))

    def almost_equal(self, other: ParacleteVec, tol: float = 1e-9) -> bool:
        if not isinstance(other, ParacleteVec):
            return False
        threshold = float(tol)
        for a, b in zip(self.data, other.data):
            if abs(a - b) > threshold:
                return False
        return True

    # ---------- Geometry / Projections ----------

    def to3(self) -> Tuple[float, float, float]:
        x = sum(self.data[0:4])
        y = sum(self.data[4:8])
        z = sum(self.data[8:12])
        return float(x), float(y), float(z)

    def project_onto_axes(self, axes: Tuple[int, ...]) -> ParacleteVec:
        buf = [0.0] * 12
        for i in axes:
            if 0 <= i < 12:
                buf[i] = self.data[i]
        return ParacleteVec(tuple(buf))

    def distance_to(self, other: ParacleteVec) -> float:
        return (self - other).norm()

    def angle_with(self, other: ParacleteVec) -> float:
        norm_self = self.norm()
        norm_other = other.norm()
        if norm_self < 1e-10 or norm_other < 1e-10:
            return 0.0
        cos_angle = self.dot(other) / (norm_self * norm_other)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    # ---------- Serialization / Display ----------

    def to_list(self) -> List[float]:
        return [float(x) for x in self.data]

    def __str__(self) -> str:
        if len(self.data) > 6:
            components = (
                f"[{self.data[0]:.3f}, {self.data[1]:.3f}, {self.data[2]:.3f}, "
                f"..., {self.data[-3]:.3f}, {self.data[-2]:.3f}, {self.data[-1]:.3f}]"
            )
        else:
            components = f"{[f'{x:.3f}' for x in self.data]}"
        return f"ParacleteVec({components})"

    def __repr__(self) -> str:
        return f"ParacleteVec(data={self.data})"
