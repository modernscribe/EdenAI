"""
Phase Tree Structures and Builders

This module defines the hierarchical phase tree structures that organize
the Paraclete state space. Includes PhaseKey, PhaseNode, PhaseTree classes
and builders for direct and inverse trees.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
import json

from vectors import ParacleteVec
from logos import (
    key_to_paraclete_vec,
    sym_hfp,
    cube_exponent,
    inv_exponent,
    anti_exponent,
    MODALITIES,
    ORIENTATIONS,
    CHARGES,
    infer_reality_scale,
    soul_state_signature,
)


@dataclass(frozen=True)
class PhaseKey:
    """
    Immutable key identifying a phase node's position in the tree.
    
    Attributes:
        depth: Tree depth (0 = root)
        branch: Path from root as tuple of modality strings
        charge: +1 or -1
        orientation: "alpha" or "omega"
        modality: "tangible", "existential", or "symbolic"
    """
    depth: int
    branch: Tuple[str, ...]
    charge: int
    orientation: str
    modality: str

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"Depth must be >= 0, got {self.depth}")
        if self.charge not in (+1, -1):
            raise ValueError(f"Charge must be +1 or -1, got {self.charge}")
        if self.orientation not in ("alpha", "omega"):
            raise ValueError(f"Invalid orientation: {self.orientation}")
        if self.modality not in ("tangible", "existential", "symbolic"):
            raise ValueError(f"Invalid modality: {self.modality}")

    def id_str(self) -> str:
        b = "/".join(self.branch) if self.branch else "∅"
        c = "+" if self.charge > 0 else "-"
        return f"d{self.depth}:{b}:{c}:{self.orientation}:{self.modality}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "branch": list(self.branch),
            "charge": self.charge,
            "orientation": self.orientation,
            "modality": self.modality,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PhaseKey:
        return PhaseKey(
            depth=data["depth"],
            branch=tuple(data["branch"]),
            charge=data["charge"],
            orientation=data["orientation"],
            modality=data["modality"],
        )


@dataclass
class PhaseNode:
    """
    A node in the phase tree with associated vector and relationships.
    
    Attributes:
        key: PhaseKey identifying this node
        exponent: Integer exponent for vector scaling
        symbol: HFP symbol representation
        inverse_id: ID of charge-inverse node (optional)
        dual_id: ID of depth-dual node (optional)
        children: List of child nodes
    """
    key: PhaseKey
    exponent: int
    symbol: str
    inverse_id: Optional[str] = None
    dual_id: Optional[str] = None
    children: List[PhaseNode] = field(default_factory=list)

    def add_child(self, child: PhaseNode) -> None:
        self.children.append(child)

    def to_vec(self) -> ParacleteVec:
        return key_to_paraclete_vec(
            self.key.modality,
            self.key.orientation,
            self.key.charge,
            self.exponent,
        )

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def descendant_count(self) -> int:
        count = len(self.children)
        for child in self.children:
            count += child.descendant_count()
        return count

    def reality_scale(self) -> str:
        vec = self.to_vec()
        try:
            scale = infer_reality_scale(vec)
            name = getattr(scale, "name", str(scale))
            return name
        except Exception:
            return "UNKNOWN"

    def soul_signature(self) -> Dict[str, Any]:
        vec = self.to_vec()
        try:
            sig = soul_state_signature(vec)
            if isinstance(sig, dict):
                return sig
            return {"signature": sig}
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "exponent": self.exponent,
            "symbol": self.symbol,
            "inverse_id": self.inverse_id,
            "dual_id": self.dual_id,
            "children": [child.to_dict() for child in self.children],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PhaseNode:
        node = PhaseNode(
            key=PhaseKey.from_dict(data["key"]),
            exponent=data["exponent"],
            symbol=data["symbol"],
            inverse_id=data.get("inverse_id"),
            dual_id=data.get("dual_id"),
        )
        for child_data in data.get("children", []):
            node.add_child(PhaseNode.from_dict(child_data))
        return node


@dataclass
class PhaseTree:
    """
    Hierarchical tree structure organizing phase nodes.
    
    Attributes:
        root: Root PhaseNode
        index: Dictionary mapping ID strings to nodes
    """
    root: PhaseNode
    index: Dict[str, PhaseNode] = field(default_factory=dict)

    def register(self, node: PhaseNode) -> None:
        self.index[node.key.id_str()] = node

    def link_inverses(self) -> None:
        table: Dict[Tuple[int, Tuple[str, ...], str, str], Dict[int, str]] = {}
        for nid, node in self.index.items():
            k = (
                node.key.depth,
                node.key.branch,
                node.key.orientation,
                node.key.modality,
            )
            table.setdefault(k, {})[node.key.charge] = nid
        for entry in table.values():
            pos = entry.get(+1)
            neg = entry.get(-1)
            if pos and neg:
                self.index[pos].inverse_id = neg
                self.index[neg].inverse_id = pos

    def link_duals(self) -> None:
        by_depth: Dict[int, List[str]] = {}
        for nid, n in self.index.items():
            by_depth.setdefault(n.key.depth, []).append(nid)
        for depth, nodes in by_depth.items():
            ln = len(nodes)
            if ln == 0:
                continue
            half = ln // 2
            for i in range(ln):
                nid = nodes[i]
                did = nodes[(i + half) % ln]
                self.index[nid].dual_id = did

    def get_by_signature(
        self,
        modality: str,
        orientation: str,
        charge: int,
        depth: int,
    ) -> List[PhaseNode]:
        result: List[PhaseNode] = []
        for n in self.index.values():
            if (
                n.key.modality == modality
                and n.key.orientation == orientation
                and n.key.charge == charge
                and n.key.depth == depth
            ):
                result.append(n)
        return result

    def all_nodes(self) -> Iterable[PhaseNode]:
        return self.index.values()

    def get_node(self, node_id: str) -> Optional[PhaseNode]:
        return self.index.get(node_id)

    def depth_levels(self) -> Dict[int, List[PhaseNode]]:
        levels: Dict[int, List[PhaseNode]] = {}
        for node in self.all_nodes():
            levels.setdefault(node.key.depth, []).append(node)
        return levels

    def group_by_reality_scale(self) -> Dict[str, List[PhaseNode]]:
        grouped: Dict[str, List[PhaseNode]] = {}
        for node in self.all_nodes():
            scale = node.reality_scale()
            grouped.setdefault(scale, []).append(node)
        return grouped

    def branch_soul_signature(self, branch: Tuple[str, ...]) -> Dict[str, Any]:
        vectors: List[ParacleteVec] = []
        for node in self.all_nodes():
            if node.key.branch == branch:
                vectors.append(node.to_vec())
        if not vectors:
            return {}
        acc = ParacleteVec.zeros()
        for v in vectors:
            acc = acc + v
        try:
            sig = soul_state_signature(acc)
            if isinstance(sig, dict):
                return sig
            return {"signature": sig}
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "index_keys": list(self.index.keys()),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PhaseTree:
        root = PhaseNode.from_dict(data["root"])
        tree = PhaseTree(root=root)

        def register_recursive(node: PhaseNode) -> None:
            tree.register(node)
            for child in node.children:
                register_recursive(child)

        register_recursive(root)
        return tree


def build_direct_3logos_tree(max_depth: int = 2, base_exponent: int = 3) -> PhaseTree:
    root_key = PhaseKey(
        depth=0,
        branch=tuple(),
        charge=+1,
        orientation="alpha",
        modality="tangible",
    )
    root = PhaseNode(
        key=root_key,
        exponent=base_exponent,
        symbol=sym_hfp(base_exponent),
    )
    tree = PhaseTree(root=root)
    tree.register(root)

    def grow(node: PhaseNode, depth: int) -> None:
        if depth >= max_depth:
            return
        nd = depth + 1
        for m in MODALITIES:
            for o in ORIENTATIONS:
                for c in CHARGES:
                    br = node.key.branch + (m,)
                    e = cube_exponent(node.exponent)
                    if o == "omega" and c < 0:
                        e = inv_exponent(e)
                    child = PhaseNode(
                        key=PhaseKey(nd, br, c, o, m),
                        exponent=e,
                        symbol=sym_hfp(e),
                    )
                    node.add_child(child)
                    tree.register(child)
                    grow(child, nd)

    grow(root, 0)
    tree.link_inverses()
    tree.link_duals()
    return tree


def build_inverse_3logos_tree(max_depth: int = 2, base_exponent: int = 3) -> PhaseTree:
    root_key = PhaseKey(
        depth=0,
        branch=tuple(),
        charge=-1,
        orientation="omega",
        modality="tangible",
    )
    root_e = anti_exponent(base_exponent)
    root = PhaseNode(
        key=root_key,
        exponent=root_e,
        symbol=sym_hfp(root_e),
    )
    tree = PhaseTree(root=root)
    tree.register(root)

    def grow(node: PhaseNode, depth: int) -> None:
        if depth >= max_depth:
            return
        nd = depth + 1
        for m in MODALITIES:
            for o in ORIENTATIONS:
                for c in CHARGES:
                    br = node.key.branch + (m,)
                    e = cube_exponent(node.exponent)
                    if o == "alpha" and c > 0:
                        e = anti_exponent(e)
                    if o == "omega" and c < 0:
                        e = inv_exponent(e)
                    child = PhaseNode(
                        key=PhaseKey(nd, br, c, o, m),
                        exponent=e,
                        symbol=sym_hfp(e),
                    )
                    node.add_child(child)
                    tree.register(child)
                    grow(child, nd)

    grow(root, 0)
    tree.link_inverses()
    tree.link_duals()
    return tree


def merge_trees(tree1: PhaseTree, tree2: PhaseTree) -> PhaseTree:
    return tree1
