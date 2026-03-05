"""
Module d'expérimentations pour le projet.
Implémente les 4 expériences demandées (E.1 à E.4).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from grid import Grid, create_simple_grid, create_medium_grid, create_hard_grid
from astar import astar, ucs, greedy_search, weighted_astar
from markov import MarkovChain
from simulation import MonteCarloSimulator
from visualization import (
    plot_grid_with_path, compare_algorithms, plot_convergence,
    plot_policy, plot_trajectories
)


def experiment_1_compare_search_algorithms() -> pd.DataFrame:
    """
    Expérience E.1: Comparaison de A* et ses variantes.
    Compare A*, UCS, Greedy et Weighted A* sur différentes grilles.
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE E.1: Comparaison des algorithmes de recherche")
    print("="*80 + "\n")
    
    # Créer différentes grilles de test
    grids = {
        'Simple (10x10)': create_simple_grid(),
        'Moyenne (20x20)': create_medium_grid(),
        'Difficile (30x30)': create_hard_grid()
    }
    
    # Algorithmes à tester
    algorithms = {
        'A*': lambda g: astar(g, g.start, g.goal),
        'UCS': lambda g: ucs(g, g.start, g.goal),
        'Greedy': lambda g: greedy_search(g, g.start, g.goal),
        'Weighted A* (w=2)': lambda g: weighted_astar(g, g.start, g.goal, weight=2.0)
    }
    
    results = []
    
    for grid_name, grid in grids.items():
        print(f"\nTest sur grille: {grid_name}")
        print("-" * 60)
        
        grid_results = {}
        
        for algo_name, algo_func in algorithms.items():
            print(f"  {algo_name}...", end=" ")
            
            try:
                stats = algo_func(grid)
                
                result = {
                    'Grille': grid_name,
                    'Algorithme': algo_name,
                    'Coût': stats.cost,
                    'Nœuds explorés': stats.nodes_expanded,
                    'Taille OPEN max': stats.max_open_size,
                    'Temps (ms)': stats.execution_time * 1000,
                    'Longueur chemin': len(stats.path)
                }
                
                results.append(result)
                grid_results[algo_name] = stats
                
                print(f"✓ Coût: {stats.cost:.1f}, Nœuds: {stats.nodes_expanded}, "
                      f"Temps: {stats.execution_time*1000:.2f}ms")
                
            except Exception as e:
                print(f"✗ Erreur: {e}")
        
        # Visualisation comparative pour cette grille
        if grid_results:
            compare_algorithms(grid, grid_results)
    
    # Créer DataFrame des résultats
    df = pd.DataFrame(results)
    
    # Sauvegarder
    df.to_csv('results_E1.csv', index=False)
    print(f"\nRésultats sauvegardés dans results_E1.csv")
    
    # Afficher tableau récapitulatif
    print("\n" + "="*80)
    print("RÉCAPITULATIF E.1")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def experiment_2_markov_convergence() -> Dict:
    """
    Expérience E.2: Étude de la convergence de Value Iteration et Policy Iteration.
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE E.2: Convergence des algorithmes de Markov")
    print("="*80 + "\n")
    
    grid = create_medium_grid()
    
    # Tester différents paramètres
    gamma_values = [0.7, 0.8, 0.9, 0.95, 0.99]
    noise_values = [0.0, 0.1, 0.2, 0.3]
    
    results = {
        'gamma_test': [],
        'noise_test': []
    }
    
    # Test 1: Impact de gamma
    print("\n--- Test 1: Impact du facteur d'actualisation (gamma) ---")
    for gamma in gamma_values:
        print(f"Gamma = {gamma}...", end=" ")
        
        mdp = MarkovChain(grid, gamma=gamma, noise=0.1)
        vi_result = mdp.value_iteration()
        
        results['gamma_test'].append({
            'gamma': gamma,
            'iterations': vi_result['iterations'],
            'time': vi_result['time'],
            'final_delta': vi_result['deltas'][-1] if vi_result['deltas'] else 0
        })
        
        print(f"Convergence en {vi_result['iterations']} itérations "
              f"({vi_result['time']:.4f}s)")
    
    # Test 2: Impact du bruit
    print("\n--- Test 2: Impact du bruit ---")
    for noise in noise_values:
        print(f"Bruit = {noise}...", end=" ")
        
        mdp = MarkovChain(grid, gamma=0.9, noise=noise)
        vi_result = mdp.value_iteration()
        
        results['noise_test'].append({
            'noise': noise,
            'iterations': vi_result['iterations'],
            'time': vi_result['time'],
            'final_delta': vi_result['deltas'][-1] if vi_result['deltas'] else 0
        })
        
        print(f"Convergence en {vi_result['iterations']} itérations "
              f"({vi_result['time']:.4f}s)")
    
    # Test 3: Comparaison VI vs PI
    print("\n--- Test 3: Value Iteration vs Policy Iteration ---")
    mdp = MarkovChain(grid, gamma=0.9, noise=0.2)
    
    vi_result = mdp.value_iteration()
    print(f"Value Iteration: {vi_result['iterations']} itérations, "
          f"{vi_result['time']:.4f}s")
    
    mdp2 = MarkovChain(grid, gamma=0.9, noise=0.2)
    pi_result = mdp2.policy_iteration()
    print(f"Policy Iteration: {pi_result['iterations']} itérations, "
          f"{pi_result['time']:.4f}s")
    
    # Visualisation de la convergence
    if vi_result['deltas']:
        plot_convergence(
            list(range(len(vi_result['deltas']))),
            vi_result['deltas'],
            title="Convergence de Value Iteration",
            xlabel="Itérations",
            ylabel="Delta (erreur maximale)"
        )
    
    # Visualiser la politique
    plot_policy(grid, mdp.policy)
    
    return results


def experiment_3_monte_carlo_robustness() -> Dict:
    """
    Expérience E.3: Évaluation de la robustesse avec Monte Carlo.
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE E.3: Robustesse via simulations Monte Carlo")
    print("="*80 + "\n")
    
    grid = create_medium_grid()
    
    # Calculer la politique optimale
    print("Calcul de la politique optimale...")
    mdp = MarkovChain(grid, gamma=0.9, noise=0.2)
    mdp.value_iteration()
    
    # Créer le simulateur
    simulator = MonteCarloSimulator(grid, mdp)
    
    # Test 1: Simulation standard
    print("\n--- Test 1: Simulations avec bruit 0.2 ---")
    stats_standard = simulator.run_simulations(
        start=grid.start,
        n_episodes=1000,
        max_steps=500,
        use_policy=True,
        verbose=True
    )
    
    # Test 2: Comparaison avec différents niveaux de bruit
    print("\n--- Test 2: Impact du bruit ---")
    noise_comparison = simulator.compare_noise_levels(
        start=grid.start,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4],
        n_episodes=500,
        max_steps=500
    )
    
    # Visualisation des trajectoires
    print("\nVisualisation de quelques trajectoires...")
    plot_trajectories(
        grid,
        stats_standard['trajectories'][:20],
        title="20 Trajectoires Stochastiques (bruit=0.2)",
        max_trajectories=20
    )
    
    # Graphique comparatif du bruit
    noise_levels = list(noise_comparison.keys())
    success_rates = [noise_comparison[n]['success_rate'] for n in noise_levels]
    avg_steps = [noise_comparison[n]['avg_steps'] for n in noise_levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(noise_levels, success_rates, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Niveau de bruit', fontsize=12)
    ax1.set_ylabel('Taux de succès (%)', fontsize=12)
    ax1.set_title('Impact du bruit sur le succès', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(noise_levels, avg_steps, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Niveau de bruit', fontsize=12)
    ax2.set_ylabel('Nombre moyen d\'étapes', fontsize=12)
    ax2.set_title('Impact du bruit sur la longueur', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'standard': stats_standard,
        'noise_comparison': noise_comparison
    }


def experiment_4_astar_vs_markov() -> pd.DataFrame:
    """
    Expérience E.4: Comparaison A* vs Markov dans environnement stochastique.
    """
    print("\n" + "="*80)
    print("EXPÉRIENCE E.4: A* vs Markov en environnement stochastique")
    print("="*80 + "\n")
    
    grid = create_medium_grid()
    
    # 1. Obtenir le chemin A*
    print("1. Calcul du chemin A*...")
    astar_stats = astar(grid, grid.start, grid.goal)
    astar_path = astar_stats.path
    print(f"   Chemin A*: {len(astar_path)} étapes, coût: {astar_stats.cost}")
    
    # 2. Calculer la politique Markov
    print("\n2. Calcul de la politique Markov...")
    mdp = MarkovChain(grid, gamma=0.9, noise=0.2)
    mdp.value_iteration()
    markov_path = mdp.get_policy_path(grid.start)
    print(f"   Chemin Markov: {len(markov_path)} étapes")
    
    # 3. Simuler avec A* dans environnement stochastique
    print("\n3. Simulation de A* en environnement stochastique...")
    
    # Créer un "MDP" qui suit le chemin A* (politique déterministe)
    astar_policy = {}
    for i in range(len(astar_path) - 1):
        astar_policy[astar_path[i]] = astar_path[i + 1]
    astar_policy[astar_path[-1]] = astar_path[-1]  # État terminal
    
    # Créer un MDP temporaire pour simulation
    astar_mdp = MarkovChain(grid, gamma=0.9, noise=0.2)
    astar_mdp.policy = astar_policy
    
    astar_simulator = MonteCarloSimulator(grid, astar_mdp)
    astar_sim_stats = astar_simulator.run_simulations(
        start=grid.start,
        n_episodes=1000,
        max_steps=500,
        use_policy=True,
        verbose=False
    )
    
    # 4. Simuler avec Markov
    print("4. Simulation de Markov en environnement stochastique...")
    markov_simulator = MonteCarloSimulator(grid, mdp)
    markov_sim_stats = markov_simulator.run_simulations(
        start=grid.start,
        n_episodes=1000,
        max_steps=500,
        use_policy=True,
        verbose=False
    )
    
    # 5. Comparaison
    print("\n" + "="*80)
    print("COMPARAISON DES RÉSULTATS")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Métrique': [
            'Taux de succès (%)',
            'Étapes moyennes',
            'Écart-type étapes',
            'Récompense moyenne',
            'Écart-type récompense'
        ],
        'A* (déterministe)': [
            astar_sim_stats['success_rate'],
            astar_sim_stats['avg_steps'],
            astar_sim_stats['std_steps'],
            astar_sim_stats['avg_reward'],
            astar_sim_stats['std_reward']
        ],
        'Markov (stochastique)': [
            markov_sim_stats['success_rate'],
            markov_sim_stats['avg_steps'],
            markov_sim_stats['std_steps'],
            markov_sim_stats['avg_reward'],
            markov_sim_stats['std_reward']
        ]
    })
    
    print(comparison.to_string(index=False))
    
    # Visualisation comparative
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1: Chemins déterministes
    grid_results = {
        'A*': astar_stats,
        'Markov': type('Stats', (), {
            'path': markov_path,
            'cost': len(markov_path) - 1,
            'nodes_expanded': 0,
            'max_open_size': 0,
            'execution_time': 0
        })()
    }
    
    for idx, (name, stats) in enumerate(grid_results.items()):
        ax = axes[idx]
        
        vis_grid = np.zeros((grid.height, grid.width, 3))
        vis_grid[:, :] = [1, 1, 1]
        
        for (x, y) in grid.obstacles:
            vis_grid[y, x] = [0, 0, 0]
        
        if stats.path:
            for (x, y) in stats.path[1:-1]:
                vis_grid[y, x] = [0.5, 1.0, 0.5]
        
        if grid.start:
            vis_grid[grid.start[1], grid.start[0]] = [1, 1, 0]
        if grid.goal:
            vis_grid[grid.goal[1], grid.goal[0]] = [1, 0, 0]
        
        ax.imshow(vis_grid, origin='upper', extent=[0, grid.width, grid.height, 0])
        ax.set_xticks(range(0, grid.width + 1, 5))
        ax.set_yticks(range(0, grid.height + 1, 5))
        ax.grid(True, color='gray', linewidth=0.5)
        ax.set_title(f'{name} - Chemin déterministe', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Graphique 2: Comparaison statistiques
    metrics = ['Taux succès\n(%)', 'Étapes\nmoyennes']
    astar_vals = [astar_sim_stats['success_rate'], astar_sim_stats['avg_steps']]
    markov_vals = [markov_sim_stats['success_rate'], markov_sim_stats['avg_steps']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, astar_vals, width, label='A*', color='skyblue')
    ax.bar(x + width/2, markov_vals, width, label='Markov', color='lightcoral')
    
    ax.set_ylabel('Valeur', fontsize=12)
    ax.set_title('Comparaison A* vs Markov en environnement stochastique',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    comparison.to_csv('results_E4.csv', index=False)
    print(f"\nRésultats sauvegardés dans results_E4.csv")
    
    return comparison


def run_all_experiments() -> None:
    """Lance toutes les expériences."""
    print("\n" + "="*80)
    print("LANCEMENT DE TOUTES LES EXPÉRIENCES")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # E.1
    print("\n[1/4] Expérience E.1...")
    df_e1 = experiment_1_compare_search_algorithms()
    
    # E.2
    print("\n[2/4] Expérience E.2...")
    results_e2 = experiment_2_markov_convergence()
    
    # E.3
    print("\n[3/4] Expérience E.3...")
    results_e3 = experiment_3_monte_carlo_robustness()
    
    # E.4
    print("\n[4/4] Expérience E.4...")
    df_e4 = experiment_4_astar_vs_markov()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TOUTES LES EXPÉRIENCES TERMINÉES")
    print("="*80)
    print(f"Temps total: {total_time:.2f}s")
    print("\nFichiers générés:")
    print("  - results_E1.csv")
    print("  - results_E4.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Menu interactif
    print("\n" + "="*80)
    print("MODULE D'EXPÉRIMENTATIONS - Mini Projet Bases IA")
    print("="*80)
    print("\nChoisissez une expérience à lancer:")
    print("  1. E.1 - Comparaison des algorithmes de recherche")
    print("  2. E.2 - Convergence des algorithmes de Markov")
    print("  3. E.3 - Robustesse via Monte Carlo")
    print("  4. E.4 - A* vs Markov en environnement stochastique")
    print("  5. Toutes les expériences")
    print("  0. Quitter")
    
    choice = input("\nVotre choix (0-5): ").strip()
    
    if choice == '1':
        experiment_1_compare_search_algorithms()
    elif choice == '2':
        experiment_2_markov_convergence()
    elif choice == '3':
        experiment_3_monte_carlo_robustness()
    elif choice == '4':
        experiment_4_astar_vs_markov()
    elif choice == '5':
        run_all_experiments()
    else:
        print("Au revoir!")
