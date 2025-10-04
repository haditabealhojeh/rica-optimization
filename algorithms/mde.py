"""
Manifold Differential Evolution (mDE) implementation - CORRECTED VERSION.
"""

import numpy as np
from typing import List, Tuple, Any
from .base_optimizer import BaseOptimizer
from manifolds.manifold_utils import scale_tangent, add_tangents


class MDEOptimizer(BaseOptimizer):
    """Manifold Differential Evolution algorithm - CORRECTED."""
    
    def __init__(self, manifold, cost_function, population_size=50, max_iterations=100,
                 F=0.5, CR=0.9, seed=None, verbose=True):
        
        super().__init__(manifold, cost_function, population_size, max_iterations, seed, verbose)
        self.F = F  # mutation factor
        self.CR = CR  # crossover rate
        
    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run mDE optimization - CORRECTED."""
        
        population = self._initialize_population()
        costs = self._evaluate_population(population)
        
        self._update_best(population, costs)
        self.convergence_history = [self.best_cost]
        
        for iteration in range(self.max_iterations):
            new_population = []
            new_costs = []
            
            for i in range(self.population_size):
                # Select three distinct random individuals - CORRECTED selection
                indices = [j for j in range(self.population_size) if j != i]
                a_idx, b_idx, c_idx = np.random.choice(indices, 3, replace=False)
                a, b, c = population[a_idx], population[b_idx], population[c_idx]
                
                # Mutation: v = a + F * (b - c) - CORRECTED direction
                try:
                    # Calculate b - c on the manifold (CORRECTED: b minus c)
                    diff_bc = self._manifold_subtract(b, c)  # This is b - c
                    # Scale the difference
                    scaled_diff = scale_tangent(diff_bc, self.F)
                    # Retract to get mutant vector v = a + F*(b-c)
                    v = self.manifold.retraction(a, scaled_diff)
                except Exception as e:
                    if self.verbose:
                        print(f"mDE mutation error: {e}. Using random point.")
                    v = self.manifold.random_point()
                
                # Crossover - CORRECTED implementation
                trial = self._crossover(population[i], v)
                
                # Selection
                try:
                    trial_cost = self.cost_function(trial)
                    current_cost = costs[i]
                    
                    if trial_cost < current_cost:
                        new_population.append(trial)
                        new_costs.append(trial_cost)
                        # Update best solution
                        if trial_cost < self.best_cost:
                            self.best_solution = trial
                            self.best_cost = trial_cost
                    else:
                        new_population.append(population[i])
                        new_costs.append(current_cost)
                        
                except Exception as e:
                    if self.verbose:
                        print(f"mDE cost evaluation error: {e}")
                    # Keep original if trial fails
                    new_population.append(population[i])
                    new_costs.append(costs[i])
            
            population = new_population
            costs = new_costs
            self._log_convergence(iteration, self.best_cost)
            self.convergence_history.append(self.best_cost)
        
        return self.best_solution, self.best_cost, self.convergence_history
    
    def _manifold_subtract(self, point1, point2):
        """Subtract two points on the manifold using logarithmic map - CORRECTED."""
        try:
            # Use logarithmic map for proper manifold subtraction: point1 - point2
            return self.manifold.log(point2, point1)  # CORRECTED: point1 - point2
        except NotImplementedError:
            # Fallback: use ambient space subtraction with projection
            if isinstance(point1, list) and isinstance(point2, list):
                # For product manifolds
                ambient_diff = [p1 - p2 for p1, p2 in zip(point1, point2)]
                return self.manifold.projection(point2, ambient_diff)
            else:
                # For single manifolds
                ambient_diff = point1 - point2
                return self.manifold.projection(point2, ambient_diff)
    
    def _crossover(self, target, mutant):
        """Perform crossover between target and mutant vectors - CORRECTED."""
        if isinstance(target, list):
            # For product manifolds - CORRECTED crossover
            trial = []
            crossover_occurred = False
            
            for t_elem, m_elem in zip(target, mutant):
                if np.random.rand() < self.CR:
                    trial.append(m_elem)
                    crossover_occurred = True
                else:
                    trial.append(t_elem)
            
            # Ensure at least one component from mutant
            if not crossover_occurred:
                idx = np.random.randint(len(target))
                trial[idx] = mutant[idx]
                
            return trial
        else:
            # For single manifolds - binomial crossover
            if np.random.rand() < self.CR:
                return mutant
            else:
                return target