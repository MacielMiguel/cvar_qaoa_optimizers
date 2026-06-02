import numpy as np
from typing import List, Tuple

from problems.maxmulticut.hamiltonian import MaxMulticutHamiltonian
from quantum_core.graphs import GraphInstanceGenerator


class MaxMulticutInstanceGenerator(GraphInstanceGenerator):
    """Generate MaxMulticut problem instances."""

    # ------------------------------------------------------------------
    # Random instances
    # ------------------------------------------------------------------

    @staticmethod
    def random(
        num_vertices: int,
        edge_density: float = 0.5,
        seed: int | None = None
    ) -> List[Tuple[int, int]]:
        """Return a random edge list with reproducible seed."""
        return GraphInstanceGenerator.random(num_vertices, edge_density, seed)  # type: ignore

    @staticmethod
    def random_instance(
        num_vertices: int,
        edge_density: float = 0.5,
        seed: int | None = None,
        num_colors: int = 3
    ) -> MaxMulticutHamiltonian:
        """Random MaxMulticut instance with reproducible seed."""
        edges = GraphInstanceGenerator.random(num_vertices, edge_density, seed)  # type: ignore
        return MaxMulticutHamiltonian(edges, num_colors=num_colors, num_vertices=num_vertices)

    @staticmethod
    def random_edges(num_vertices: int, edge_density: float = 0.5, seed: int | None = None) -> List[Tuple[int, int]]:
        """Return only the edge list for compatibility with GraphInstanceGenerator.random signature."""
        return MaxMulticutInstanceGenerator.random(num_vertices, edge_density, seed)

    @staticmethod
    def complete_instance(num_vertices: int, num_colors: int = 3) -> MaxMulticutHamiltonian:
        """Complete graph (all edges present)."""
        edges = GraphInstanceGenerator.complete(num_vertices)
        return MaxMulticutHamiltonian(edges, num_colors=num_colors, num_vertices=num_vertices)

    @staticmethod
    def grid_instance(rows: int, cols: int, num_colors: int = 3) -> MaxMulticutHamiltonian:
        """Grid graph."""
        edges = GraphInstanceGenerator.grid(rows, cols)
        num_vertices = rows * cols
        return MaxMulticutHamiltonian(edges, num_colors=num_colors, num_vertices=num_vertices)

    @staticmethod
    def bipartite_instance(num_vertices: int, num_colors: int = 3, partition_ratio: float = 0.5) -> MaxMulticutHamiltonian:
        """Bipartite graph (edges only between two partitions)."""
        edges = GraphInstanceGenerator.bipartite(num_vertices, partition_ratio)
        return MaxMulticutHamiltonian(edges, num_colors=num_colors, num_vertices=num_vertices)

    @staticmethod
    def k_partite_instance(
        num_vertices: int,
        num_partitions: int,
        num_colors: int = 3,
        inter_partition_density: float = 0.5,
        seed: int | None = None
    ) -> MaxMulticutHamiltonian:
        """k-partite graph: vertices divided into k groups, edges only between groups."""
        edges = GraphInstanceGenerator.k_partite(
            num_vertices, 
            num_partitions, 
            inter_partition_density, 
            seed if seed is not None else 0
        )
        return MaxMulticutHamiltonian(edges, num_colors=num_colors, num_vertices=num_vertices)

    # ------------------------------------------------------------------
    # Fixed benchmark instances
    # ------------------------------------------------------------------

    @staticmethod
    def fixed_simple(num_colors: int = 3) -> MaxMulticutHamiltonian:
        """Simple 6-vertex fixed instance."""
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)]
        return MaxMulticutHamiltonian(edges, num_colors=num_colors, num_vertices=6)

    @staticmethod
    def fixed_grid_3x3(num_colors: int = 3) -> MaxMulticutHamiltonian:
        """3x3 grid graph."""
        return MaxMulticutInstanceGenerator.grid_instance(3, 3, num_colors=num_colors)

    @staticmethod
    def fixed_grid_4x4(num_colors: int = 3) -> MaxMulticutHamiltonian:
        """4x4 grid graph."""
        return MaxMulticutInstanceGenerator.grid_instance(4, 4, num_colors=num_colors)
