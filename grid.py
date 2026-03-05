"""
Module de définition de la grille et de l'environnement.
Gère les obstacles, les positions de départ et d'arrivée.
"""

import numpy as np
from typing import List, Tuple, Set, Optional

class Grid:
    """Représente une grille 2D avec obstacles."""
    
    def __init__(self, width: int, height: int):
        """
        Initialise une grille vide.
        
        Args:
            width: Largeur de la grille
            height: Hauteur de la grille
        """
        self.width = width
        self.height = height
        self.obstacles: Set[Tuple[int, int]] = set()
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.cost_map: dict = {}  # Coûts variables (optionnel)
        
    def add_obstacle(self, x: int, y: int) -> None:
        """Ajoute un obstacle à la position (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))
    
    def add_obstacles(self, obstacles: List[Tuple[int, int]]) -> None:
        """Ajoute plusieurs obstacles."""
        for obs in obstacles:
            self.add_obstacle(obs[0], obs[1])
    
    def add_rectangular_obstacle(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Ajoute un obstacle rectangulaire."""
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                self.add_obstacle(x, y)
    
    def set_start(self, x: int, y: int) -> None:
        """Définit la position de départ."""
        self.start = (x, y)
    
    def set_goal(self, x: int, y: int) -> None:
        """Définit la position d'arrivée."""
        self.goal = (x, y)
    
    def is_valid(self, x: int, y: int) -> bool:
        """Vérifie si une position est valide (dans la grille et sans obstacle)."""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retourne les voisins valides d'une position (4-voisinage).
        
        Args:
            pos: Position actuelle (x, y)
            
        Returns:
            Liste des positions voisines valides
        """
        x, y = pos
        neighbors = []
        # Ordre: droite, bas, gauche, haut
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def get_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Retourne le coût pour se déplacer de from_pos à to_pos.
        
        Args:
            from_pos: Position de départ
            to_pos: Position d'arrivée
            
        Returns:
            Coût du déplacement (par défaut 1.0)
        """
        # Coût variable si défini, sinon coût uniforme de 1
        return self.cost_map.get((from_pos, to_pos), 1.0)
    
    def set_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], cost: float) -> None:
        """Définit un coût spécifique pour un déplacement."""
        self.cost_map[(from_pos, to_pos)] = cost
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance euclidienne entre deux positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def __str__(self) -> str:
        """Représentation textuelle de la grille."""
        grid_str = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == self.start:
                    row.append('S')
                elif (x, y) == self.goal:
                    row.append('G')
                elif (x, y) in self.obstacles:
                    row.append('#')
                else:
                    row.append('.')
            grid_str.append(''.join(row))
        return '\n'.join(grid_str)


def create_simple_grid() -> Grid:
    """Crée une grille simple de test (facile)."""
    grid = Grid(10, 10)
    grid.set_start(1, 1)
    grid.set_goal(8, 8)
    # Quelques obstacles dispersés
    grid.add_obstacles([(3, 3), (3, 4), (3, 5), (6, 2), (6, 3), (6, 4)])
    return grid


def create_medium_grid() -> Grid:
    """Crée une grille de difficulté moyenne."""
    grid = Grid(15, 15)
    grid.set_start(1, 1)
    grid.set_goal(13, 13)
    
    # Murs plus complexes
    grid.add_rectangular_obstacle(5, 0, 5, 8)
    grid.add_rectangular_obstacle(10, 6, 10, 14)
    grid.add_obstacles([(7, 10), (8, 10), (9, 10)])
    grid.add_obstacles([(3, 12), (3, 13), (4, 12)])
    
    return grid


def create_hard_grid() -> Grid:
    """Crée une grille difficile avec un labyrinthe."""
    grid = Grid(20, 20)
    grid.set_start(1, 1)
    grid.set_goal(18, 18)
    
    # Labyrinthe complexe
    grid.add_rectangular_obstacle(3, 0, 3, 15)
    grid.add_rectangular_obstacle(6, 5, 6, 19)
    grid.add_rectangular_obstacle(9, 0, 9, 14)
    grid.add_rectangular_obstacle(12, 6, 12, 19)
    grid.add_rectangular_obstacle(15, 0, 15, 15)
    
    # Ouvertures
    grid.obstacles.discard((3, 2))
    grid.obstacles.discard((6, 7))
    grid.obstacles.discard((9, 12))
    grid.obstacles.discard((12, 8))
    grid.obstacles.discard((15, 13))
    
    return grid


if __name__ == "__main__":
    # Test des grilles
    print("=== Grille Simple ===")
    simple = create_simple_grid()
    print(simple)
    
    print("\n=== Grille Moyenne ===")
    medium = create_medium_grid()
    print(medium)
    
    print("\n=== Grille Difficile ===")
    hard = create_hard_grid()
    print(hard)
