from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import math

from vectors import ParacleteVec
from phase_tree import PhaseTree, PhaseNode


@dataclass(frozen=True)
class PhaseContext:
    plane: ParacleteVec
    slice_vec: ParacleteVec
    phase_vec: ParacleteVec

    @property
    def combined(self) -> ParacleteVec:
        return self.plane + self.slice_vec + self.phase_vec

    @staticmethod
    def from_tree(tree: PhaseTree) -> PhaseContext:
        tang = tree.get_by_signature("tangible", "alpha", +1, 1)
        exist = tree.get_by_signature("existential", "alpha", +1, 1)
        symb = tree.get_by_signature("symbolic", "alpha", +1, 1)
        tang_src = tang[0] if tang else tree.root
        exist_src = exist[0] if exist else tree.root
        symb_src = symb[0] if symb else tree.root
        plane = tang_src.to_vec().normalized()
        slice_vec = exist_src.to_vec().normalized()
        phase_vec = symb_src.to_vec().normalized()
        return PhaseContext(plane=plane, slice_vec=slice_vec, phase_vec=phase_vec)

    @staticmethod
    def zero() -> PhaseContext:
        zero = ParacleteVec.zeros()
        return PhaseContext(plane=zero, slice_vec=zero, phase_vec=zero)

    @staticmethod
    def from_vectors(
        plane: ParacleteVec,
        slice_vec: ParacleteVec,
        phase_vec: ParacleteVec,
        normalize: bool = True,
    ) -> PhaseContext:
        if normalize:
            plane = plane.normalized()
            slice_vec = slice_vec.normalized()
            phase_vec = phase_vec.normalized()
        return PhaseContext(plane=plane, slice_vec=slice_vec, phase_vec=phase_vec)

    def perturb(
        self,
        plane_delta: Optional[ParacleteVec] = None,
        slice_delta: Optional[ParacleteVec] = None,
        phase_delta: Optional[ParacleteVec] = None,
        normalize: bool = True,
    ) -> PhaseContext:
        new_plane = self.plane + (plane_delta if plane_delta is not None else ParacleteVec.zeros())
        new_slice = self.slice_vec + (slice_delta if slice_delta is not None else ParacleteVec.zeros())
        new_phase = self.phase_vec + (phase_delta if phase_delta is not None else ParacleteVec.zeros())
        if normalize:
            new_plane = new_plane.normalized()
            new_slice = new_slice.normalized()
            new_phase = new_phase.normalized()
        return PhaseContext(plane=new_plane, slice_vec=new_slice, phase_vec=new_phase)

    def rotate_plane(self, angle: float, axis1: int, axis2: int) -> PhaseContext:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        plane_data = list(self.plane.data)
        old1 = plane_data[axis1]
        old2 = plane_data[axis2]
        plane_data[axis1] = cos_a * old1 - sin_a * old2
        plane_data[axis2] = sin_a * old1 + cos_a * old2
        new_plane = ParacleteVec.from_list(plane_data).normalized()
        return PhaseContext(plane=new_plane, slice_vec=self.slice_vec, phase_vec=self.phase_vec)

    def animate(
        self,
        time: float,
        plane_func: Optional[Callable[[float], ParacleteVec]] = None,
        slice_func: Optional[Callable[[float], ParacleteVec]] = None,
        phase_func: Optional[Callable[[float], ParacleteVec]] = None,
    ) -> PhaseContext:
        new_plane = plane_func(time) if plane_func is not None else self.plane
        new_slice = slice_func(time) if slice_func is not None else self.slice_vec
        new_phase = phase_func(time) if phase_func is not None else self.phase_vec
        return PhaseContext(
            plane=new_plane.normalized(),
            slice_vec=new_slice.normalized(),
            phase_vec=new_phase.normalized(),
        )

    def interpolate(self, other: PhaseContext, t: float) -> PhaseContext:
        t = max(0.0, min(1.0, t))
        plane = (1.0 - t) * self.plane + t * other.plane
        slice_vec = (1.0 - t) * self.slice_vec + t * other.slice_vec
        phase_vec = (1.0 - t) * self.phase_vec + t * other.phase_vec
        return PhaseContext(
            plane=plane.normalized(),
            slice_vec=slice_vec.normalized(),
            phase_vec=phase_vec.normalized(),
        )

    def apply_to_node(self, node: PhaseNode) -> ParacleteVec:
        return node.to_vec() + self.combined

    def to_dict(self) -> dict:
        return {
            "plane": self.plane.to_list(),
            "slice_vec": self.slice_vec.to_list(),
            "phase_vec": self.phase_vec.to_list(),
        }

    @staticmethod
    def from_dict(data: dict) -> PhaseContext:
        return PhaseContext(
            plane=ParacleteVec.from_list(data["plane"]),
            slice_vec=ParacleteVec.from_list(data["slice_vec"]),
            phase_vec=ParacleteVec.from_list(data["phase_vec"]),
        )


class ContextAnimator:
    def __init__(self, base_context: PhaseContext):
        self.base = base_context

    def circular_plane(self, time: float, frequency: float = 1.0) -> ParacleteVec:
        angle = 2.0 * math.pi * frequency * time
        return self.base.rotate_plane(angle, 0, 1).plane

    def oscillating_slice(
        self,
        time: float,
        frequency: float = 1.0,
        amplitude: float = 0.1,
    ) -> ParacleteVec:
        scale = 1.0 + amplitude * math.sin(2.0 * math.pi * frequency * time)
        return scale * self.base.slice_vec

    def spiral_phase(
        self,
        time: float,
        frequency: float = 1.0,
        growth: float = 0.01,
    ) -> ParacleteVec:
        angle = 2.0 * math.pi * frequency * time
        scale = 1.0 + growth * time
        rotated = self.base.rotate_plane(angle, 4, 5).phase_vec
        return scale * rotated
