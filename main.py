#!/usr/bin/env python3
"""
Script principal pour exécuter rapidement le projet.
"""

import sys

def main():
    print("=" * 70)
    print(" " * 15 + "MINI-PROJET: PLANIFICATION ROBUSTE")
    print(" " * 20 + "A* + Chaînes de Markov")
    print("=" * 70)
    print()
    
    print("Choisissez une option:")
    print("  1. Exécuter toutes les expériences (E.1 à E.4)")
    print("  2. Test rapide (grille simple)")
    print("  3. Lancer le notebook Jupyter")
    print("  4. Quitter")
    print()
    
    choice = input("Votre choix (1-4): ").strip()
    
    if choice == "1":
        print("\n→ Exécution de toutes les expériences...\n")
        from experiments import run_all_experiments
        run_all_experiments()
    
    elif choice == "2":
        print("\n→ Test rapide sur grille simple...\n")
        quick_test()
    
    elif choice == "3":
        print("\n→ Lancement du notebook Jupyter...\n")
        import os
        os.system("jupyter notebook notebook.ipynb")
    
    elif choice == "4":
        print("\n→ Au revoir!\n")
        sys.exit(0)
    
    else:
        print("\n❌ Choix invalide. Veuillez choisir 1, 2, 3 ou 4.\n")
        main()


def quick_test():
    """Test rapide sur une grille simple."""
    from grid import create_simple_grid
    from astar import PathFinder
    from markov import MarkovChain
    from simulation import MonteCarloSimulator
    from visualization import Visualizer
    
    print("1. Création d'une grille simple...")
    grid = create_simple_grid()
    print(grid)
    
    print("\n2. Recherche de chemin avec A*...")
    finder = PathFinder(grid)
    stats = finder.astar()
    
    if stats.success:
        print(f"   ✓ Chemin trouvé!")
        print(f"   Coût: {stats.cost:.2f}")
        print(f"   Longueur: {len(stats.path)}")
        print(f"   Nœuds développés: {stats.nodes_expanded}")
        print(f"   Temps: {stats.execution_time:.4f}s")
        
        print("\n3. Construction de la chaîne de Markov (ε=0.15)...")
        markov = MarkovChain(grid, stats.path, epsilon=0.15)
        analysis = markov.analyze()
        print(f"   ✓ Matrice P: {analysis.transition_matrix.shape}")
        print(f"   États absorbants: {len(analysis.absorbing_states)}")
        
        print("\n4. Simulation Monte-Carlo (1000 trajectoires)...")
        simulator = MonteCarloSimulator(grid, markov)
        results = simulator.run_simulations(n_simulations=1000)
        print(f"   ✓ Taux de succès: {results.success_rate:.2%}")
        print(f"   Temps moyen: {results.mean_time_to_goal:.2f} étapes")
        
        print("\n5. Visualisation...")
        vis = Visualizer()
        vis.plot_grid_with_path(grid, stats.path, title="Test Rapide: Chemin A*")
        
        print("\n✓ Test rapide terminé avec succès!")
    else:
        print("   ❌ Aucun chemin trouvé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n→ Interruption par l'utilisateur. Au revoir!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")
        sys.exit(1)
