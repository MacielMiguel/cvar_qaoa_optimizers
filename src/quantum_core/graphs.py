"""
Base class for graph instance generation.
Shared utilities for creating graph structures used in combinatorial optimization problems.
"""

import numpy as np
from typing import List, Tuple


class GraphInstanceGenerator:
    """Base class with shared graph generation utilities."""

    @staticmethod
    def random(num_vertices: int, edge_density: float = 0.5, seed: int = None) -> List[Tuple[int, int]]:
        """Generate random edges with reproducible seed.
        
        Args:
            num_vertices: Number of vertices.
            edge_density: Probability of edge between any two vertices (0 < density < 1).
            seed: Random seed for reproducibility.
        
        Returns:
            List of (u, v) edge tuples.
        """
        rng = np.random.RandomState(seed)
        edges = [
            (i, j)
            for i in range(num_vertices)
            for j in range(i + 1, num_vertices)
            if rng.random() < edge_density
        ]
        return edges

    @staticmethod
    def complete(num_vertices: int) -> List[Tuple[int, int]]:
        """Generate complete graph (all edges present).
        
        Args:
            num_vertices: Number of vertices.
        
        Returns:
            List of all (u, v) edge tuples.
        """
        edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)]
        return edges

    @staticmethod
    def grid(rows: int, cols: int) -> List[Tuple[int, int]]:
        """Generate grid graph edges.
        
        Args:
            rows: Number of rows.
            cols: Number of columns.
        
        Returns:
            List of (u, v) edge tuples forming a grid.
        """
        edges = []
        for i in range(rows):
            for j in range(cols):
                v = i * cols + j
                if j < cols - 1:
                    edges.append((v, v + 1))
                if i < rows - 1:
                    edges.append((v, v + cols))
        return edges

    @staticmethod
    def bipartite(num_vertices: int, partition_ratio: float = 0.5) -> List[Tuple[int, int]]:
        """Generate bipartite graph edges (edges only between two partitions).
        
        Args:
            num_vertices: Total number of vertices.
            partition_ratio: Ratio determining partition sizes (0 < ratio < 1).
        
        Returns:
            List of (u, v) edge tuples between partitions.
        """
        k = int(num_vertices * partition_ratio)
        edges = [(i, j) for i in range(k) for j in range(k, num_vertices)]
        return edges

    @staticmethod
    def k_partite(
        num_vertices: int,
        num_partitions: int,
        inter_partition_density: float = 0.5,
        seed: int = None
    ) -> List[Tuple[int, int]]:
        """Generate k-partite graph edges (edges only between different partitions).
        
        Args:
            num_vertices: Total number of vertices.
            num_partitions: Number of partitions.
            inter_partition_density: Probability of edge between different partitions.
            seed: Random seed.
        
        Returns:
            List of (u, v) edge tuples between partitions.
        """
        rng = np.random.RandomState(seed)
        vertices_per_partition = num_vertices // num_partitions
        partitions = [
            list(range(p * vertices_per_partition, (p + 1) * vertices_per_partition))
            for p in range(num_partitions - 1)
        ]
        # Last partition gets remaining vertices
        partitions.append(list(range((num_partitions - 1) * vertices_per_partition, num_vertices)))

        edges = []
        for p1 in range(num_partitions):
            for p2 in range(p1 + 1, num_partitions):
                for u in partitions[p1]:
                    for v in partitions[p2]:
                        if rng.random() < inter_partition_density:
                            edges.append((min(u, v), max(u, v)))

        return edges
