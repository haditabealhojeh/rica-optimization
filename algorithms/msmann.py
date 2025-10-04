"""
Manifold Simulated Annealing (mSMANN) implementation - CORRECTED VERSION.
"""

import numpy as np
from typing import List, Tuple, Any
from .base_optimizer import BaseOptimizer
from manifolds.manifold_utils import scale_tangent, random_tangent_vector


class MSMANNOptimizer(BaseOptimizer):
    """Manifold Simulated Annealing algorithm - CORRECTED."""
    
    def __init__(self, manifold, cost_function, population_size=50, max_iterations=100,
                 initial_temperature=100.0, cooling_rate=0.95, 
                 step_size=1.0, seed=None, verbose=True):
        
        super().__init__(manifold, cost_function, population_size, max_iterations, seed, verbose)
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        
    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run mSMANN optimization - CORRECTED."""
        
        # For simulated annealing, we maintain population but track single best
        population = self._initialize_population()
        costs = self._evaluate_population(population)
        
        self._update_best(population, costs)
        self.convergence_history = [self.best_cost]
        
        # Main annealing loop - CORRECTED temperature schedule
        for iteration in range(self.max_iterations):
            new_population = []
            new_costs = []
            
            for i in range(self.population_size):
                current_point = population[i]
                current_cost = costs[i]
                
                # Generate neighbor using manifold-aware random walk
                neighbor = self._generate_neighbor(current_point, iteration)
                neighbor_cost = self.cost_function(neighbor)
                
                # Metropolis acceptance criterion - CORRECTED
                if self._accept_solution(current_cost, neighbor_cost):
                    new_population.append(neighbor)
                    new_costs.append(neighbor_cost)
                else:
                    new_population.append(current_point)
                    new_costs.append(current_cost)
            
            population = new_population
            costs = new_costs
            self._update_best(population, costs)
            
            # Cool down temperature - CORRECTED schedule
            self.temperature = self.initial_temperature * np.power(self.cooling_rate, iteration)
            
            self._log_convergence(iteration, self.best_cost)
            self.convergence_history.append(self.best_cost)
        
        return self.best_solution, self.best_cost, self.convergence_history
    
    def _generate_neighbor(self, point, iteration):
        """Generate a neighboring point on the manifold - CORRECTED."""
        try:
            # Generate random tangent vector with temperature-dependent scale
            current_temp = self.initial_temperature * np.power(self.cooling_rate, iteration)
            temperature_factor = np.sqrt(current_temp / self.initial_temperature)
            
            tangent_step = random_tangent_vector(self.manifold, point, scale=self.step_size)
            
            # Add temperature-dependent random noise - CORRECTED scaling
            step_scale = temperature_factor * np.random.normal(0, 1.0)
            noisy_step = scale_tangent(tangent_step, step_scale)
            
            # Retract to get new point on manifold
            neighbor = self.manifold.retraction(point, noisy_step)
            return neighbor
            
        except Exception as e:
            if self.verbose:
                print(f"mSMANN neighbor generation error: {e}. Using random point.")
            return self.manifold.random_point()
    
    def _accept_solution(self, current_cost, new_cost):
        """Determine whether to accept new solution - CORRECTED Metropolis criterion."""
        if new_cost < current_cost:
            return True
        else:
            # Calculate acceptance probability
            delta_cost = new_cost - current_cost
            acceptance_prob = np.exp(-delta_cost / (self.temperature + 1e-12))
            return np.random.rand() < acceptance_prob