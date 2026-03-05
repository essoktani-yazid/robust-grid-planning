"""
Module d'implémentation de A* et ses variantes (UCS, Greedy).
Inclut les structures OPEN (heap) et CLOSED (set).
"""

import heapq
import time
from typing import Tuple, List, Dict, Optional, Callable, Set
from dataclasses import dataclass, field
from grid import Grid


@dataclass(order=True)
class Node:
    """Représente un nœud dans l'arbre de recherche."""
    f: float  # f(n) = g(n) + h(n)
    position: Tuple[int, int] = field(compare=False)
    g: float = field(compare=False)  # Coût depuis le départ
    h: float = field(compare=False)  # Heuristique vers le but
    parent: Optional[Tuple[int, int]] = field(default=None, compare=False)


@dataclass
class SearchStats:
    """Statistiques de recherche."""
    path: List[Tuple[int, int]]
    cost: float
    nodes_expanded: int
    max_open_size: int
    execution_time: float
    success: bool


class PathFinder:
    """Classe pour les algorithmes de recherche de chemin."""
    
    def __init__(self, grid: Grid):
        """
        Initialise le chercheur de chemin.
        
        Args:
            grid: Grille de navigation
        """
        self.grid = grid
    
    def manhattan_heuristic(self, pos: Tuple[int, int]) -> float:
        """Heuristique de Manhattan (admissible pour 4-voisinage)."""
        if self.grid.goal is None:
            return 0.0
        return self.grid.manhattan_distance(pos, self.grid.goal)
    
    def euclidean_heuristic(self, pos: Tuple[int, int]) -> float:
        """Heuristique euclidienne."""
        if self.grid.goal is None:
            return 0.0
        return self.grid.euclidean_distance(pos, self.grid.goal)
    
    def zero_heuristic(self, pos: Tuple[int, int]) -> float:
        """Heuristique nulle (équivalent à UCS)."""
        return 0.0
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruit le chemin depuis la position courante vers le départ."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def astar(self, 
              heuristic: Optional[Callable[[Tuple[int, int]], float]] = None,
              weight: float = 1.0) -> SearchStats:
        """
        Algorithme A* (et variantes).
        
        Args:
            heuristic: Fonction heuristique (défaut: Manhattan)
            weight: Poids de l'heuristique (1.0 = A*, >1 = Weighted A*, 0 = UCS)
            
        Returns:
            SearchStats avec le chemin et les statistiques
        """
        if self.grid.start is None or self.grid.goal is None:
            raise ValueError("Start et goal doivent être définis")
        
        if heuristic is None:
            heuristic = self.manhattan_heuristic
        
        start_time = time.time()
        
        # Structures de données
        open_set: List[Node] = []
        closed_set: Set[Tuple[int, int]] = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {self.grid.start: 0.0}
        
        # Initialisation
        h_start = heuristic(self.grid.start)
        start_node = Node(
            f=weight * h_start,
            position=self.grid.start,
            g=0.0,
            h=h_start,
            parent=None
        )
        heapq.heappush(open_set, start_node)
        
        nodes_expanded = 0
        max_open_size = 1
        
        while open_set:
            max_open_size = max(max_open_size, len(open_set))
            
            # Extraire le nœud avec le plus petit f
            current_node = heapq.heappop(open_set)
            current = current_node.position
            
            # But atteint
            if current == self.grid.goal:
                path = self.reconstruct_path(came_from, current)
                execution_time = time.time() - start_time
                return SearchStats(
                    path=path,
                    cost=current_node.g,
                    nodes_expanded=nodes_expanded,
                    max_open_size=max_open_size,
                    execution_time=execution_time,
                    success=True
                )
            
            # Ignorer si déjà visité
            if current in closed_set:
                continue
            
            closed_set.add(current)
            nodes_expanded += 1
            
            # Explorer les voisins
            for neighbor in self.grid.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculer le coût
                tentative_g = current_node.g + self.grid.get_cost(current, neighbor)
                
                # Si c'est un meilleur chemin
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    
                    h_neighbor = heuristic(neighbor)
                    f_neighbor = tentative_g + weight * h_neighbor
                    
                    neighbor_node = Node(
                        f=f_neighbor,
                        position=neighbor,
                        g=tentative_g,
                        h=h_neighbor,
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)
        
        # Pas de chemin trouvé
        execution_time = time.time() - start_time
        return SearchStats(
            path=[],
            cost=float('inf'),
            nodes_expanded=nodes_expanded,
            max_open_size=max_open_size,
            execution_time=execution_time,
            success=False
        )
    
    def ucs(self) -> SearchStats:
        """
        Uniform Cost Search (UCS) = A* avec h(n) = 0.
        
        Returns:
            SearchStats avec le chemin et les statistiques
        """
        return self.astar(heuristic=self.zero_heuristic, weight=0.0)
    
    def greedy(self, heuristic: Optional[Callable[[Tuple[int, int]], float]] = None) -> SearchStats:
        """
        Greedy Best-First Search: f(n) = h(n).
        
        Args:
            heuristic: Fonction heuristique (défaut: Manhattan)
            
        Returns:
            SearchStats avec le chemin et les statistiques
        """
        if heuristic is None:
            heuristic = self.manhattan_heuristic
        
        if self.grid.start is None or self.grid.goal is None:
            raise ValueError("Start et goal doivent être définis")
        
        start_time = time.time()
        
        # Structures de données
        open_set: List[Node] = []
        closed_set: Set[Tuple[int, int]] = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {self.grid.start: 0.0}
        
        # Initialisation (f = h uniquement pour Greedy)
        h_start = heuristic(self.grid.start)
        start_node = Node(
            f=h_start,  # Greedy: utilise seulement h
            position=self.grid.start,
            g=0.0,
            h=h_start,
            parent=None
        )
        heapq.heappush(open_set, start_node)
        
        nodes_expanded = 0
        max_open_size = 1
        
        while open_set:
            max_open_size = max(max_open_size, len(open_set))
            
            current_node = heapq.heappop(open_set)
            current = current_node.position
            
            if current == self.grid.goal:
                path = self.reconstruct_path(came_from, current)
                execution_time = time.time() - start_time
                return SearchStats(
                    path=path,
                    cost=g_score[current],  # Coût réel du chemin
                    nodes_expanded=nodes_expanded,
                    max_open_size=max_open_size,
                    execution_time=execution_time,
                    success=True
                )
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            nodes_expanded += 1
            
            for neighbor in self.grid.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self.grid.get_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    
                    h_neighbor = heuristic(neighbor)
                    
                    neighbor_node = Node(
                        f=h_neighbor,  # Greedy: f = h seulement
                        position=neighbor,
                        g=tentative_g,
                        h=h_neighbor,
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)
        
        execution_time = time.time() - start_time
        return SearchStats(
            path=[],
            cost=float('inf'),
            nodes_expanded=nodes_expanded,
            max_open_size=max_open_size,
            execution_time=execution_time,
            success=False
        )
    
    def weighted_astar(self, weight: float = 1.5) -> SearchStats:
        """
        Weighted A*: f(n) = g(n) + w*h(n) avec w > 1.
        Compromis entre vitesse et optimalité.
        
        Args:
            weight: Poids de l'heuristique (typiquement 1.5-2.0)
            
        Returns:
            SearchStats avec le chemin et les statistiques
        """
        return self.astar(weight=weight)


# Fonctions helper pour compatibilité avec experiments.py
def astar(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchStats:
    """Fonction wrapper pour A*."""
    grid.set_start(start[0], start[1])
    grid.set_goal(goal[0], goal[1])
    finder = PathFinder(grid)
    return finder.astar()


def ucs(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchStats:
    """Fonction wrapper pour UCS."""
    grid.set_start(start[0], start[1])
    grid.set_goal(goal[0], goal[1])
    finder = PathFinder(grid)
    return finder.ucs()


def greedy_search(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchStats:
    """Fonction wrapper pour Greedy."""
    grid.set_start(start[0], start[1])
    grid.set_goal(goal[0], goal[1])
    finder = PathFinder(grid)
    return finder.greedy()


def weighted_astar(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int], weight: float = 1.5) -> SearchStats:
    """Fonction wrapper pour Weighted A*."""
    grid.set_start(start[0], start[1])
    grid.set_goal(goal[0], goal[1])
    finder = PathFinder(grid)
    return finder.weighted_astar(weight)


if __name__ == "__main__":
    from grid import create_simple_grid, create_medium_grid
    
    # Test sur grille simple
    print("=== Test A* sur grille simple ===")
    grid = create_simple_grid()
    print(grid)
    print()
    
    finder = PathFinder(grid)
    
    # Comparer UCS, Greedy et A*
    print("--- UCS ---")
    stats_ucs = finder.ucs()
    print(f"Chemin trouvé: {stats_ucs.success}")
    print(f"Coût: {stats_ucs.cost}")
    print(f"Nœuds développés: {stats_ucs.nodes_expanded}")
    print(f"Temps: {stats_ucs.execution_time:.4f}s")
    print(f"Longueur chemin: {len(stats_ucs.path)}")
    
    print("\n--- Greedy ---")
    stats_greedy = finder.greedy()
    print(f"Chemin trouvé: {stats_greedy.success}")
    print(f"Coût: {stats_greedy.cost}")
    print(f"Nœuds développés: {stats_greedy.nodes_expanded}")
    print(f"Temps: {stats_greedy.execution_time:.4f}s")
    print(f"Longueur chemin: {len(stats_greedy.path)}")
    
    print("\n--- A* ---")
    stats_astar = finder.astar()
    print(f"Chemin trouvé: {stats_astar.success}")
    print(f"Coût: {stats_astar.cost}")
    print(f"Nœuds développés: {stats_astar.nodes_expanded}")
    print(f"Temps: {stats_astar.execution_time:.4f}s")
    print(f"Longueur chemin: {len(stats_astar.path)}")
    print(f"Chemin: {stats_astar.path}")
