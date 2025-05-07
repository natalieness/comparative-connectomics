

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Union, Any


@dataclass
class NeuronLevelChange:
    operation: str
    source_index: int
    new_index: int
    weight_handling: str


@dataclass
class EdgeLevelChange:
    operation: str
    position: Tuple[int, int]
    old_value: float
    new_value: float

class ChangeLog:
    def __init__(self):
        self.neuron_level_changes: List[NeuronLevelChange] = []
        self.edge_level_changes: List[EdgeLevelChange] = []

    def add_neuron_change(self, operation: str, source_index: int, new_index: int, weight_handling: str):
        self.neuron_level_changes.append(
            NeuronLevelChange(operation, source_index, new_index, weight_handling)
        )

    def add_edge_change(self, operation: str, position: Tuple[int, int], old_value: float, new_value: float):
        self.edge_level_changes.append(
            EdgeLevelChange(operation, position, old_value, new_value)
        )

    def search_by_operation(self, operation: str) -> Dict[str, List[Dict[str, Any]]]:
        results = {
            "neuron-level": [asdict(c) for c in self.neuron_level_changes if c.operation == operation],
            "edge-level": [asdict(c) for c in self.edge_level_changes if c.operation == operation],
        }
        return results

    def __repr__(self):
        out = ["Neuron-Level Changes:"]
        for c in self.neuron_level_changes:
            out.append(f"  {asdict(c)}")
        out.append("Edge-Level Changes:")
        for c in self.edge_level_changes:
            out.append(f"  {asdict(c)}")
        return "\n".join(out)