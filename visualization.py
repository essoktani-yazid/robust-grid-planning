"""
Module de visualisation de la grille et des chemins.
Utilise matplotlib pour afficher les résultats.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple, Optional, Dict
from grid import Grid
from astar import SearchStats


def plot_grid_with_path(
    grid: Grid, 
    path: Optional[List[Tuple[int, int]]] = None,
    title: str = "Grille avec Chemin",
    explored: Optional[set] = None,
    show_costs: bool = False
) -> None:
    """
    Visualise la grille avec obstacles et chemin.
    
    Args:
        grid: La grille à visualiser
        path: Le chemin trouvé (liste de positions)
        title: Titre du graphique
        explored: Ensemble des nœuds explorés (pour visualisation)
        show_costs: Afficher les coûts sur chaque case
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Créer une matrice pour la visualisation
    vis_grid = np.zeros((grid.height, grid.width, 3))
    
    # Fond blanc
    vis_grid[:, :] = [1, 1, 1]
    
    # Obstacles en noir
    for (x, y) in grid.obstacles:
        vis_grid[y, x] = [0, 0, 0]
    
    # Nœuds explorés en bleu clair
    if explored:
        for (x, y) in explored:
            if (x, y) not in grid.obstacles:
                vis_grid[y, x] = [0.7, 0.9, 1.0]
    
    # Chemin en vert
    if path:
        for (x, y) in path[1:-1]:  # Exclure start et goal
            vis_grid[y, x] = [0.5, 1.0, 0.5]
    
    # Position de départ en jaune
    if grid.start:
        vis_grid[grid.start[1], grid.start[0]] = [1, 1, 0]
    
    # Position d'arrivée en rouge
    if grid.goal:
        vis_grid[grid.goal[1], grid.goal[0]] = [1, 0, 0]
    
    # Afficher la grille
    ax.imshow(vis_grid, origin='upper', extent=[0, grid.width, grid.height, 0])
    
    # Grille
    ax.set_xticks(range(grid.width + 1))
    ax.set_yticks(range(grid.height + 1))
    ax.grid(True, color='gray', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Légende
    legend_elements = [
        mpatches.Patch(color='yellow', label='Départ'),
        mpatches.Patch(color='red', label='Arrivée'),
        mpatches.Patch(color='black', label='Obstacles'),
        mpatches.Patch(color='lightgreen', label='Chemin'),
    ]
    if explored:
        legend_elements.append(mpatches.Patch(color='lightblue', label='Explorés'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Afficher les coûts si demandé
    if show_costs and grid.cost_map:
        for (x, y), cost in grid.cost_map.items():
            if (x, y) not in grid.obstacles:
                ax.text(x + 0.5, y + 0.5, f'{cost:.1f}', 
                       ha='center', va='center', fontsize=8, color='blue')
    
    plt.tight_layout()
    plt.show()


def compare_algorithms(
    grid: Grid,
    results: Dict[str, SearchStats],
    save_path: Optional[str] = None
) -> None:
    """
    Compare visuellement plusieurs algorithmes.
    
    Args:
        grid: La grille utilisée
        results: Dictionnaire {nom_algo: SearchStats}
        save_path: Chemin pour sauvegarder la figure
    """
    n_algos = len(results)
    fig, axes = plt.subplots(1, n_algos, figsize=(6*n_algos, 6))
    
    if n_algos == 1:
        axes = [axes]
    
    for ax, (algo_name, stats) in zip(axes, results.items()):
        # Créer matrice de visualisation
        vis_grid = np.zeros((grid.height, grid.width, 3))
        vis_grid[:, :] = [1, 1, 1]
        
        # Obstacles
        for (x, y) in grid.obstacles:
            vis_grid[y, x] = [0, 0, 0]
        
        # Chemin
        if stats.path:
            for (x, y) in stats.path[1:-1]:
                vis_grid[y, x] = [0.5, 1.0, 0.5]
        
        # Start et Goal
        if grid.start:
            vis_grid[grid.start[1], grid.start[0]] = [1, 1, 0]
        if grid.goal:
            vis_grid[grid.goal[1], grid.goal[0]] = [1, 0, 0]
        
        # Afficher
        ax.imshow(vis_grid, origin='upper', extent=[0, grid.width, grid.height, 0])
        ax.set_xticks(range(grid.width + 1))
        ax.set_yticks(range(grid.height + 1))
        ax.grid(True, color='gray', linewidth=0.5)
        
        # Titre avec stats
        title = f"{algo_name}\n"
        title += f"Coût: {stats.cost:.1f} | "
        title += f"Nœuds: {stats.nodes_expanded} | "
        title += f"Temps: {stats.execution_time*1000:.2f}ms"
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence(
    iterations: List[int],
    values: List[float],
    title: str = "Convergence",
    xlabel: str = "Itérations",
    ylabel: str = "Valeur"
) -> None:
    """
    Graphique de convergence pour les algorithmes itératifs.
    
    Args:
        iterations: Liste des numéros d'itérations
        values: Liste des valeurs correspondantes
        title: Titre du graphique
        xlabel: Label de l'axe X
        ylabel: Label de l'axe Y
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, values, 'b-', linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_policy(grid: Grid, policy: Dict[Tuple[int, int], Tuple[int, int]]) -> None:
    """
    Visualise une politique (Markov) avec des flèches.
    
    Args:
        grid: La grille
        policy: Dictionnaire {état: action_suivante}
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Créer matrice de visualisation
    vis_grid = np.zeros((grid.height, grid.width, 3))
    vis_grid[:, :] = [1, 1, 1]
    
    # Obstacles
    for (x, y) in grid.obstacles:
        vis_grid[y, x] = [0, 0, 0]
    
    # Goal
    if grid.goal:
        vis_grid[grid.goal[1], grid.goal[0]] = [1, 0, 0]
    
    # Start
    if grid.start:
        vis_grid[grid.start[1], grid.start[0]] = [1, 1, 0]
    
    ax.imshow(vis_grid, origin='upper', extent=[0, grid.width, grid.height, 0])
    
    # Dessiner les flèches de la politique
    for (x, y), (next_x, next_y) in policy.items():
        if (x, y) != (next_x, next_y):  # Pas de flèche pour état terminal
            dx = next_x - x
            dy = next_y - y
            ax.arrow(x + 0.5, y + 0.5, dx * 0.3, dy * 0.3,
                    head_width=0.15, head_length=0.1, fc='blue', ec='blue',
                    linewidth=2)
    
    ax.set_xticks(range(grid.width + 1))
    ax.set_yticks(range(grid.height + 1))
    ax.grid(True, color='gray', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Politique Optimale', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_trajectories(
    grid: Grid,
    trajectories: List[List[Tuple[int, int]]],
    title: str = "Trajectoires Stochastiques",
    max_trajectories: int = 10
) -> None:
    """
    Visualise plusieurs trajectoires sur la grille.
    
    Args:
        grid: La grille
        trajectories: Liste de trajectoires (chaque trajectoire est une liste de positions)
        title: Titre du graphique
        max_trajectories: Nombre maximum de trajectoires à afficher
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Créer matrice de visualisation
    vis_grid = np.zeros((grid.height, grid.width, 3))
    vis_grid[:, :] = [1, 1, 1]
    
    # Obstacles
    for (x, y) in grid.obstacles:
        vis_grid[y, x] = [0, 0, 0]
    
    # Goal
    if grid.goal:
        vis_grid[grid.goal[1], grid.goal[0]] = [1, 0, 0]
    
    # Start
    if grid.start:
        vis_grid[grid.start[1], grid.start[0]] = [1, 1, 0]
    
    ax.imshow(vis_grid, origin='upper', extent=[0, grid.width, grid.height, 0])
    
    # Dessiner les trajectoires
    colors = plt.cm.rainbow(np.linspace(0, 1, min(len(trajectories), max_trajectories)))
    
    for i, traj in enumerate(trajectories[:max_trajectories]):
        if len(traj) > 1:
            x_coords = [pos[0] + 0.5 for pos in traj]
            y_coords = [pos[1] + 0.5 for pos in traj]
            ax.plot(x_coords, y_coords, 'o-', color=colors[i], 
                   alpha=0.6, linewidth=1.5, markersize=4)
    
    ax.set_xticks(range(grid.width + 1))
    ax.set_yticks(range(grid.height + 1))
    ax.grid(True, color='gray', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test basique
    grid = Grid(10, 10)
    grid.set_start(1, 1)
    grid.set_goal(8, 8)
    grid.add_rectangular_obstacle(3, 3, 5, 5)
    
    # Test de visualisation
    test_path = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (5, 3), 
                 (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)]
    
    plot_grid_with_path(grid, test_path, "Test de Visualisation")
