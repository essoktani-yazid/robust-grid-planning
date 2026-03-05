"""
Module de simulation Monte Carlo pour évaluer la robustesse.
Simule des trajectoires stochastiques et calcule des statistiques.
"""

import numpy as np
from typing import Tuple, List, Dict
from grid import Grid
from markov import MarkovChain
import time


class MonteCarloSimulator:
    """
    Simulateur Monte Carlo pour évaluer les politiques dans un environnement stochastique.
    """
    
    def __init__(self, grid: Grid, mdp: MarkovChain):
        """
        Initialise le simulateur.
        
        Args:
            grid: La grille de l'environnement
            mdp: Le MDP avec la politique à évaluer
        """
        self.grid = grid
        self.mdp = mdp
    
    def simulate_episode(
        self,
        start: Tuple[int, int],
        max_steps: int = 500,
        use_policy: bool = True
    ) -> Dict:
        """
        Simule un épisode complet.
        
        Args:
            start: Position de départ
            max_steps: Nombre maximum d'étapes
            use_policy: Utiliser la politique du MDP (sinon random)
            
        Returns:
            Dict avec 'trajectory', 'reward', 'success', 'steps'
        """
        trajectory = [start]
        current = start
        total_reward = 0
        success = False
        
        for step in range(max_steps):
            if current == self.grid.goal:
                success = True
                break
            
            # Choisir l'action
            if use_policy:
                # Suivre la politique mais avec bruit
                intended_next = self.mdp.policy.get(current, current)
                
                # Calculer l'action correspondante
                action = (intended_next[0] - current[0], intended_next[1] - current[1])
                
                # Appliquer le bruit
                if np.random.random() < self.mdp.noise and action != (0, 0):
                    # Choisir une action perpendiculaire
                    if action in [(0, 1), (0, -1)]:
                        perp_actions = [(1, 0), (-1, 0)]
                    else:
                        perp_actions = [(0, 1), (0, -1)]
                    action = perp_actions[np.random.randint(2)]
            else:
                # Action aléatoire
                action = self.mdp.actions[np.random.randint(len(self.mdp.actions))]
            
            # Appliquer l'action
            next_state = self.mdp.get_next_state(current, action)
            
            # Calculer la récompense
            reward = self.mdp.rewards[next_state]
            total_reward += reward
            
            # Mettre à jour
            trajectory.append(next_state)
            current = next_state
        
        return {
            'trajectory': trajectory,
            'reward': total_reward,
            'success': success,
            'steps': len(trajectory) - 1
        }
    
    def run_simulations(
        self,
        start: Tuple[int, int],
        n_episodes: int = 1000,
        max_steps: int = 500,
        use_policy: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Lance plusieurs simulations et calcule des statistiques.
        
        Args:
            start: Position de départ
            n_episodes: Nombre d'épisodes à simuler
            max_steps: Nombre maximum d'étapes par épisode
            use_policy: Utiliser la politique du MDP
            verbose: Afficher la progression
            
        Returns:
            Dict avec statistiques agrégées
        """
        start_time = time.time()
        
        results = {
            'trajectories': [],
            'rewards': [],
            'successes': [],
            'steps': []
        }
        
        for episode in range(n_episodes):
            if verbose and episode % 100 == 0:
                print(f"Épisode {episode}/{n_episodes}...")
            
            result = self.simulate_episode(start, max_steps, use_policy)
            
            results['trajectories'].append(result['trajectory'])
            results['rewards'].append(result['reward'])
            results['successes'].append(result['success'])
            results['steps'].append(result['steps'])
        
        # Calculer les statistiques
        stats = {
            'n_episodes': n_episodes,
            'success_rate': np.mean(results['successes']) * 100,
            'avg_reward': np.mean(results['rewards']),
            'std_reward': np.std(results['rewards']),
            'avg_steps': np.mean(results['steps']),
            'std_steps': np.std(results['steps']),
            'min_steps': np.min(results['steps']),
            'max_steps': np.max(results['steps']),
            'execution_time': time.time() - start_time,
            'trajectories': results['trajectories']  # Garder pour visualisation
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Résultats de la simulation Monte Carlo ({n_episodes} épisodes)")
            print(f"{'='*60}")
            print(f"Taux de succès: {stats['success_rate']:.2f}%")
            print(f"Récompense moyenne: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"Nombre d'étapes moyen: {stats['avg_steps']:.2f} ± {stats['std_steps']:.2f}")
            print(f"Nombre d'étapes [min, max]: [{stats['min_steps']}, {stats['max_steps']}]")
            print(f"Temps d'exécution: {stats['execution_time']:.2f}s")
            print(f"{'='*60}\n")
        
        return stats
    
    def compare_noise_levels(
        self,
        start: Tuple[int, int],
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
        n_episodes: int = 500,
        max_steps: int = 500
    ) -> Dict:
        """
        Compare les performances pour différents niveaux de bruit.
        
        Args:
            start: Position de départ
            noise_levels: Liste des niveaux de bruit à tester
            n_episodes: Nombre d'épisodes par niveau
            max_steps: Nombre maximum d'étapes
            
        Returns:
            Dict avec résultats pour chaque niveau de bruit
        """
        results = {}
        
        for noise in noise_levels:
            print(f"\n--- Test avec bruit = {noise} ---")
            
            # Créer un nouveau MDP avec ce niveau de bruit
            mdp = MarkovChain(self.grid, gamma=self.mdp.gamma, noise=noise)
            mdp.values = self.mdp.values.copy()  # Garder les mêmes valeurs
            mdp._extract_policy()  # Recalculer la politique
            
            # Créer un simulateur temporaire
            temp_sim = MonteCarloSimulator(self.grid, mdp)
            
            # Lancer les simulations
            stats = temp_sim.run_simulations(
                start, n_episodes, max_steps, use_policy=True, verbose=False
            )
            
            results[noise] = stats
            
            print(f"Taux de succès: {stats['success_rate']:.2f}%")
            print(f"Étapes moyennes: {stats['avg_steps']:.2f}")
        
        return results
    
    def analyze_path_deviation(
        self,
        start: Tuple[int, int],
        optimal_path: List[Tuple[int, int]],
        n_episodes: int = 100
    ) -> Dict:
        """
        Analyse la déviation par rapport au chemin optimal.
        
        Args:
            start: Position de départ
            optimal_path: Chemin optimal de référence
            n_episodes: Nombre d'épisodes à analyser
            
        Returns:
            Dict avec statistiques de déviation
        """
        deviations = []
        
        for _ in range(n_episodes):
            result = self.simulate_episode(start, max_steps=500, use_policy=True)
            trajectory = result['trajectory']
            
            # Calculer la déviation (nombre de cases différentes)
            deviation = 0
            for i, pos in enumerate(trajectory):
                if i < len(optimal_path):
                    if pos != optimal_path[i]:
                        deviation += 1
                else:
                    deviation += 1
            
            deviations.append(deviation)
        
        return {
            'avg_deviation': np.mean(deviations),
            'std_deviation': np.std(deviations),
            'min_deviation': np.min(deviations),
            'max_deviation': np.max(deviations)
        }


if __name__ == "__main__":
    # Test basique
    from grid import Grid
    from markov import MarkovChain
    
    print("Création de l'environnement...")
    grid = Grid(10, 10)
    grid.set_start(1, 1)
    grid.set_goal(8, 8)
    grid.add_rectangular_obstacle(3, 3, 6, 6)
    
    print("Calcul de la politique optimale...")
    mdp = MarkovChain(grid, gamma=0.9, noise=0.2)
    mdp.value_iteration()
    
    print("Lancement des simulations Monte Carlo...")
    simulator = MonteCarloSimulator(grid, mdp)
    stats = simulator.run_simulations(
        start=grid.start,
        n_episodes=1000,
        max_steps=500,
        use_policy=True,
        verbose=True
    )
