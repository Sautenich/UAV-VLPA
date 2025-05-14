"""
A* Pathfinding Algorithm Implementation

This module implements the A* pathfinding algorithm for UAV path planning.
The algorithm finds the optimal path between two points while considering
obstacles and minimizing the total path cost.
"""

from typing import Dict, List, Set, Tuple, Optional
import math


class Graph:
    """A graph representation for A* pathfinding."""
    
    def __init__(self, adjacency_list: Dict[int, List[Tuple[int, float]]]):
        """
        Initialize the graph with an adjacency list.
        
        Args:
            adjacency_list: Dictionary mapping vertices to their neighbors and edge weights.
                          Format: {vertex: [(neighbor, weight), ...]}
        """
        self.adjacency_list = adjacency_list

    def get_neighbors(self, vertex: int) -> List[Tuple[int, float]]:
        """
        Get all neighbors of a vertex.
        
        Args:
            vertex: The vertex to get neighbors for.
            
        Returns:
            List of tuples containing (neighbor_vertex, edge_weight).
        """
        return self.adjacency_list[vertex]

    def calculate_heuristic(self, current: int, goal: int, points: List[Tuple[float, float]]) -> float:
        """
        Calculate the heuristic value (estimated cost) from current node to goal.
        Uses Euclidean distance as the heuristic.
        
        Args:
            current: Current vertex index
            goal: Goal vertex index
            points: List of (x, y) coordinates for all vertices
            
        Returns:
            Estimated cost from current to goal
        """
        return math.sqrt(
            (points[goal][0] - points[current][0]) ** 2 +
            (points[goal][1] - points[current][1]) ** 2
        )

    def find_path(self, points: List[Tuple[float, float]], goal_node: int) -> List[int]:
        """
        Find the optimal path using A* algorithm.
        
        Args:
            points: List of (x, y) coordinates for all vertices
            goal_node: Index of the goal vertex
            
        Returns:
            List of vertex indices representing the optimal path
        """
        start_node = goal_node - 1
        open_set: Set[int] = {start_node}
        closed_set: Set[int] = set()
        
        # g_scores contains current distances from start_node to all other nodes
        g_scores: Dict[int, float] = {start_node: 0}
        
        # parents contains the path reconstruction information
        parents: Dict[int, int] = {start_node: start_node}
        
        while open_set:
            # Find node with lowest f_score (g_score + heuristic)
            current = min(
                open_set,
                key=lambda v: g_scores.get(v, float('inf')) + 
                    self.calculate_heuristic(v, goal_node, points)
            )
            
            # Check if we've reached the goal
            if current == goal_node:
                return self._reconstruct_path(parents, current, start_node)
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor, weight in self.get_neighbors(current):
                if neighbor > goal_node:
                    continue
                    
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_scores[current] + weight
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_scores.get(neighbor, float('inf')):
                    continue
                
                # This path is the best until now. Record it!
                parents[neighbor] = current
                g_scores[neighbor] = tentative_g_score
        
        print('No path exists!')
        return []

    def _reconstruct_path(self, parents: Dict[int, int], current: int, start: int) -> List[int]:
        """
        Reconstruct the path from the parents dictionary.
        
        Args:
            parents: Dictionary mapping each node to its parent in the optimal path
            current: Current (goal) node
            start: Start node
            
        Returns:
            List of nodes in the path from start to goal
        """
        path = []
        while parents[current] != current:
            path.append(current)
            current = parents[current]
        path.append(start)
        return list(reversed(path))