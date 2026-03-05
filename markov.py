"""
Module d'implémentation des chaînes de Markov pour la planification robuste.
Inclut Value Iteration et Policy Iteration.
"""

import numpy as np
from typing import Tuple, List, Dict, Set, Optional
from grid import Grid
import time


class MarkovChain:
    """
    Implémente un Markov Decision Process (MDP) pour la navigation dans une grille.
    """
    
    def __init__(self, grid: Grid, gamma: float = 0.9, noise: float = 0.2):
        """
        Initialise le MDP.
        
        Args:
            grid: La grille de l'environnement
            gamma: Facteur d'actualisation (discount factor)
            noise: Probabilité d'erreur dans les transitions (0 = déterministe)
        """
        self.grid = grid
        self.gamma = gamma
        self.noise = noise
        
        # Générer l'espace d'états (toutes les cases non-obstacles)
        self.states = self._generate_states()
        
        # Actions possibles (4-voisinage)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Bas, Haut, Droite, Gauche
        
        # Initialisation
        self.values = {s: 0.0 for s in self.states}
        self.policy = {s: (0, 0) for s in self.states}  # Politique initiale
        
        # Récompenses
        self.rewards = self._initialize_rewards()
    
    def _generate_states(self) -> Set[Tuple[int, int]]:
        """Génère tous les états valides (cases non-obstacles)."""
        states = set()
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if (x, y) not in self.grid.obstacles:
                    states.add((x, y))
        return states
    
    def _initialize_rewards(self) -> Dict[Tuple[int, int], float]:
        """Initialise les récompenses pour chaque état."""
        rewards = {}
        for state in self.states:
            if state == self.grid.goal:
                rewards[state] = 100.0  # Récompense élevée pour le but
            elif state in self.grid.obstacles:
                rewards[state] = -100.0  # Pénalité pour obstacles
            else:
                rewards[state] = -1.0  # Coût de déplacement
        return rewards
    
    def get_next_state(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        """
        Retourne l'état suivant après application d'une action.
        Si l'état résultant est invalide, reste sur place.
        """
        next_x = state[0] + action[0]
        next_y = state[1] + action[1]
        next_state = (next_x, next_y)
        
        # Vérifier validité
        if (0 <= next_x < self.grid.width and 
            0 <= next_y < self.grid.height and
            next_state not in self.grid.obstacles):
            return next_state
        else:
            return state  # Reste sur place si invalide
    
    def get_transition_prob(
        self,
        state: Tuple[int, int],
        action: Tuple[int, int],
        next_state: Tuple[int, int]
    ) -> float:
        """
        Retourne P(s'|s,a) - probabilité de transition.
        Avec bruit: probabilité (1-noise) d'aller dans la bonne direction,
        et noise/2 d'aller perpendiculairement.
        """
        if state == self.grid.goal:
            # État terminal
            return 1.0 if next_state == state else 0.0
        
        # État résultant de l'action voulue
        intended_state = self.get_next_state(state, action)
        
        if self.noise == 0:
            # Transitions déterministes
            return 1.0 if next_state == intended_state else 0.0
        
        # Transitions stochastiques
        # Actions perpendiculaires
        perp_actions = []
        if action in [(0, 1), (0, -1)]:  # Vertical
            perp_actions = [(1, 0), (-1, 0)]
        else:  # Horizontal
            perp_actions = [(0, 1), (0, -1)]
        
        perp_states = [self.get_next_state(state, a) for a in perp_actions]
        
        prob = 0.0
        if next_state == intended_state:
            prob += (1 - self.noise)
        
        for perp_state in perp_states:
            if next_state == perp_state:
                prob += self.noise / 2
        
        return prob
    
    def value_iteration(self, theta: float = 0.01, max_iterations: int = 1000) -> Dict:
        """
        Algorithme Value Iteration pour calculer V*(s).
        
        Args:
            theta: Seuil de convergence
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Dict avec 'values', 'policy', 'iterations', 'deltas', 'time'
        """
        start_time = time.time()
        deltas = []
        
        for iteration in range(max_iterations):
            delta = 0
            new_values = self.values.copy()
            
            for state in self.states:
                if state == self.grid.goal:
                    # État terminal
                    new_values[state] = self.rewards[state]
                    continue
                
                # Calculer V(s) pour chaque action
                action_values = []
                for action in self.actions:
                    value = 0
                    for next_state in self.states:
                        prob = self.get_transition_prob(state, action, next_state)
                        if prob > 0:
                            reward = self.rewards[next_state]
                            value += prob * (reward + self.gamma * self.values[next_state])
                    action_values.append(value)
                
                # Prendre le max
                new_values[state] = max(action_values)
                delta = max(delta, abs(new_values[state] - self.values[state]))
            
            self.values = new_values
            deltas.append(delta)
            
            if delta < theta:
                break
        
        # Extraire la politique optimale
        self._extract_policy()
        
        return {
            'values': self.values.copy(),
            'policy': self.policy.copy(),
            'iterations': iteration + 1,
            'deltas': deltas,
            'time': time.time() - start_time
        }
    
    def policy_iteration(self, theta: float = 0.01, max_iterations: int = 100) -> Dict:
        """
        Algorithme Policy Iteration.
        
        Args:
            theta: Seuil de convergence pour l'évaluation
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Dict avec 'values', 'policy', 'iterations', 'time'
        """
        start_time = time.time()
        
        # Initialiser avec une politique aléatoire
        for state in self.states:
            if state != self.grid.goal:
                self.policy[state] = self.actions[0]
        
        for iteration in range(max_iterations):
            # 1. Policy Evaluation
            self._policy_evaluation(theta)
            
            # 2. Policy Improvement
            policy_stable = self._policy_improvement()
            
            if policy_stable:
                break
        
        return {
            'values': self.values.copy(),
            'policy': self.policy.copy(),
            'iterations': iteration + 1,
            'time': time.time() - start_time
        }
    
    def _policy_evaluation(self, theta: float = 0.01) -> None:
        """Évalue la politique actuelle."""
        while True:
            delta = 0
            new_values = self.values.copy()
            
            for state in self.states:
                if state == self.grid.goal:
                    continue
                
                action = self.policy[state]
                value = 0
                
                for next_state in self.states:
                    prob = self.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.rewards[next_state]
                        value += prob * (reward + self.gamma * self.values[next_state])
                
                new_values[state] = value
                delta = max(delta, abs(new_values[state] - self.values[state]))
            
            self.values = new_values
            
            if delta < theta:
                break
    
    def _policy_improvement(self) -> bool:
        """Améliore la politique. Retourne True si stable."""
        policy_stable = True
        
        for state in self.states:
            if state == self.grid.goal:
                continue
            
            old_action = self.policy[state]
            
            # Trouver la meilleure action
            best_action = old_action
            best_value = float('-inf')
            
            for action in self.actions:
                value = 0
                for next_state in self.states:
                    prob = self.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.rewards[next_state]
                        value += prob * (reward + self.gamma * self.values[next_state])
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def _extract_policy(self) -> None:
        """Extrait la politique optimale à partir des valeurs."""
        for state in self.states:
            if state == self.grid.goal:
                self.policy[state] = (0, 0)  # Pas d'action pour état terminal
                continue
            
            best_action = self.actions[0]
            best_value = float('-inf')
            
            for action in self.actions:
                value = 0
                for next_state in self.states:
                    prob = self.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.rewards[next_state]
                        value += prob * (reward + self.gamma * self.values[next_state])
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # Convertir action en état suivant pour la politique
            next_state = self.get_next_state(state, best_action)
            self.policy[state] = next_state
    
    def get_policy_path(self, start: Tuple[int, int], max_steps: int = 100) -> List[Tuple[int, int]]:
        """
        Suit la politique pour obtenir un chemin depuis start.
        
        Args:
            start: Position de départ
            max_steps: Nombre maximum d'étapes
            
        Returns:
            Liste de positions formant le chemin
        """
        path = [start]
        current = start
        
        for _ in range(max_steps):
            if current == self.grid.goal:
                break
            
            next_state = self.policy.get(current, current)
            if next_state == current:
                break
            
            path.append(next_state)
            current = next_state
        
        return path


if __name__ == "__main__":
    # Test basique
    from grid import Grid
    
    grid = Grid(10, 10)
    grid.set_start(1, 1)
    grid.set_goal(8, 8)
    grid.add_rectangular_obstacle(3, 3, 6, 6)
    
    mdp = MarkovChain(grid, gamma=0.9, noise=0.1)
    
    print("Test de Value Iteration...")
    result = mdp.value_iteration()
    print(f"Convergence en {result['iterations']} itérations")
    print(f"Temps: {result['time']:.4f}s")
    
    path = mdp.get_policy_path(grid.start)
    print(f"Chemin trouvé: {len(path)} étapes")
