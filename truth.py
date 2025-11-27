"""
Truth Table Lifting to 12D Paraclete Space

This module handles the mapping of boolean truth tables into the 12D
Paraclete vector space, treating truth values as positions with associated
temperatures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional
import itertools

from vectors import ParacleteVec
from context import PhaseContext
from phase_tree import PhaseTree


@dataclass
class TruthVariable:
    """
    A boolean variable with associated base vector in 12D.
    
    Attributes:
        name: Variable name (e.g., "A", "B", "C")
        base_vec: Base vector in 12D space
    """
    name: str
    base_vec: ParacleteVec


@dataclass
class TruthRow:
    """
    A row in a boolean truth table.
    
    Attributes:
        inputs: Dict mapping variable names to boolean values
        output: Boolean output value
    """
    inputs: Dict[str, bool]
    output: bool
    
    def to_string(self) -> str:
        """Format as readable string."""
        input_str = ", ".join(f"{k}={v}" for k, v in sorted(self.inputs.items()))
        return f"{input_str} -> {self.output}"


@dataclass
class ParacleteTruthRow:
    """
    A truth table row lifted to 12D Paraclete space.
    
    Attributes:
        inputs: Dict mapping variable names to 12D vectors
        output: Output vector in 12D
        context: PhaseContext used for lifting
        metadata: Optional metadata dict
    """
    inputs: Dict[str, ParacleteVec]
    output: ParacleteVec
    context: PhaseContext
    metadata: Dict = field(default_factory=dict)
    
    @property
    def temperature(self) -> float:
        """Get temperature (norm) of output vector."""
        return self.output.norm()
    
    @property
    def position_3d(self) -> tuple[float, float, float]:
        """Get 3D projection of output vector."""
        return self.output.to3()


@dataclass
class ParacleteTruthTable:
    """
    A complete truth table lifted to 12D space.
    
    Attributes:
        variables: Dict of TruthVariable objects
        rows: List of TruthRow objects
        context: PhaseContext for lifting
    """
    variables: Dict[str, TruthVariable]
    rows: List[TruthRow]
    context: PhaseContext

    def lift(self) -> List[ParacleteTruthRow]:
        """
        Lift boolean truth table to 12D Paraclete space.
        
        For each row:
        1. Map input variables to signed vectors based on truth values
        2. Sum input vectors to get accumulator
        3. Apply output sign to get final output vector
        
        Returns:
            List of ParacleteTruthRow objects
        """
        lifted: List[ParacleteTruthRow] = []
        ctx_vec = self.context.combined
        
        for row in self.rows:
            par_inputs: Dict[str, ParacleteVec] = {}
            accum = ParacleteVec.zeros(12)
            
            # Process input variables
            for name, value in row.inputs.items():
                var = self.variables[name]
                sign = 1.0 if value else -1.0
                v = sign * var.base_vec + ctx_vec
                par_inputs[name] = v
                accum = accum + v
            
            # Apply output sign
            out_sign = 1.0 if row.output else -1.0
            out_vec = out_sign * accum
            
            # Create lifted row with metadata
            metadata = {
                "boolean_inputs": row.inputs.copy(),
                "boolean_output": row.output,
                "accumulator_norm": accum.norm(),
            }
            
            lifted.append(
                ParacleteTruthRow(
                    inputs=par_inputs,
                    output=out_vec,
                    context=self.context,
                    metadata=metadata,
                )
            )
        
        return lifted

    def statistics(self, lifted: Optional[List[ParacleteTruthRow]] = None) -> Dict:
        """
        Compute statistics for the lifted truth table.
        
        Args:
            lifted: Pre-computed lifted rows (computed if None)
            
        Returns:
            Dictionary of statistics
        """
        if lifted is None:
            lifted = self.lift()
        
        temps = [row.temperature for row in lifted]
        true_temps = [row.temperature for row in lifted if row.metadata["boolean_output"]]
        false_temps = [row.temperature for row in lifted if not row.metadata["boolean_output"]]
        
        return {
            "num_variables": len(self.variables),
            "num_rows": len(self.rows),
            "avg_temperature": sum(temps) / len(temps) if temps else 0,
            "min_temperature": min(temps) if temps else 0,
            "max_temperature": max(temps) if temps else 0,
            "avg_true_temperature": sum(true_temps) / len(true_temps) if true_temps else 0,
            "avg_false_temperature": sum(false_temps) / len(false_temps) if false_temps else 0,
        }


def build_demo_variables_from_tree(tree: PhaseTree) -> Dict[str, TruthVariable]:
    """
    Build demo truth variables A, B, C from phase tree nodes.
    
    Args:
        tree: PhaseTree to extract nodes from
        
    Returns:
        Dict mapping variable names to TruthVariable objects
    """
    # Use specific nodes for each variable
    tang_nodes = tree.get_by_signature("tangible", "omega", -1, 0)
    exist_nodes = tree.get_by_signature("existential", "omega", -1, 1)
    symb_nodes = tree.get_by_signature("symbolic", "omega", -1, 1)
    
    a_src = tang_nodes[0] if tang_nodes else tree.root
    b_src = exist_nodes[0] if exist_nodes else tree.root
    c_src = symb_nodes[0] if symb_nodes else tree.root
    
    var_a = TruthVariable("A", a_src.to_vec())
    var_b = TruthVariable("B", b_src.to_vec())
    var_c = TruthVariable("C", c_src.to_vec())
    
    return {"A": var_a, "B": var_b, "C": var_c}


def build_truth_rows_from_function(
    variables: List[str],
    func: Callable[..., bool]
) -> List[TruthRow]:
    """
    Build truth rows from a boolean function.
    
    Args:
        variables: List of variable names
        func: Boolean function taking len(variables) boolean args
        
    Returns:
        List of TruthRow objects for all input combinations
    """
    rows: List[TruthRow] = []
    
    for values in itertools.product([False, True], repeat=len(variables)):
        inputs = dict(zip(variables, values))
        output = func(*values)
        rows.append(TruthRow(inputs, output))
    
    return rows


def build_and_truth_rows() -> List[TruthRow]:
    """Build truth table for 3-input AND."""
    return build_truth_rows_from_function(
        ["A", "B", "C"],
        lambda a, b, c: a and b and c
    )


def build_or_truth_rows() -> List[TruthRow]:
    """Build truth table for 3-input OR."""
    return build_truth_rows_from_function(
        ["A", "B", "C"],
        lambda a, b, c: a or b or c
    )


def build_majority_truth_rows() -> List[TruthRow]:
    """Build truth table for 3-input majority function."""
    return build_truth_rows_from_function(
        ["A", "B", "C"],
        lambda a, b, c: sum([a, b, c]) >= 2
    )


def build_xor_truth_rows() -> List[TruthRow]:
    """Build truth table for 3-input XOR."""
    return build_truth_rows_from_function(
        ["A", "B", "C"],
        lambda a, b, c: (a ^ b ^ c)
    )


def build_implies_truth_rows() -> List[TruthRow]:
    """Build truth table for A implies (B or C)."""
    return build_truth_rows_from_function(
        ["A", "B", "C"],
        lambda a, b, c: (not a) or (b or c)
    )


class TruthTableBuilder:
    """Helper class for building custom truth tables."""
    
    def __init__(self, tree: PhaseTree, context: PhaseContext):
        """
        Initialize builder with tree and context.
        
        Args:
            tree: PhaseTree for variable extraction
            context: PhaseContext for lifting
        """
        self.tree = tree
        self.context = context
        self.variables = build_demo_variables_from_tree(tree)
    
    def build(
        self,
        func: Callable[..., bool],
        var_names: Optional[List[str]] = None
    ) -> ParacleteTruthTable:
        """
        Build a truth table from a boolean function.
        
        Args:
            func: Boolean function
            var_names: Variable names (defaults to ["A", "B", "C"])
            
        Returns:
            ParacleteTruthTable ready for lifting
        """
        if var_names is None:
            var_names = ["A", "B", "C"]
        
        rows = build_truth_rows_from_function(var_names, func)
        vars_dict = {name: self.variables[name] for name in var_names}
        
        return ParacleteTruthTable(vars_dict, rows, self.context)
    
    def build_and(self) -> ParacleteTruthTable:
        """Build AND truth table."""
        return self.build(lambda a, b, c: a and b and c)
    
    def build_or(self) -> ParacleteTruthTable:
        """Build OR truth table."""
        return self.build(lambda a, b, c: a or b or c)
    
    def build_majority(self) -> ParacleteTruthTable:
        """Build majority truth table."""
        return self.build(lambda a, b, c: sum([a, b, c]) >= 2)
    
    def build_xor(self) -> ParacleteTruthTable:
        """Build XOR truth table."""
        return self.build(lambda a, b, c: a ^ b ^ c)
