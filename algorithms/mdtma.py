
"""
Manifold Directional Transport Metaheuristic Algorithm (mDTMA) - FIXED NO EARLY STOPPING.
"""

import numpy as np
from typing import List, Tuple, Any
from .base_optimizer import BaseOptimizer
from manifolds.manifold_utils import scale_tangent, add_tangents, zero_vector

class MDTMAOptimizer(BaseOptimizer):
    """Manifold Directional Transport Metaheuristic Algorithm - NO EARLY STOPPING."""

    def __init__(self, manifold, cost_function, population_size=50, max_iterations=100,
                 w0=0.1, crossover_rate=0.8, mutation_rate=0.1, 
                 tournament_size=2, seed=None, verbose=True):
        
        super().__init__(manifold, cost_function, population_size, max_iterations, seed, verbose)
        self.w0 = w0  # initial step size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run mDTMA optimization - FIXED: NO EARLY STOPPING."""
        
        population = self._initialize_population()
        costs = self._evaluate_population(population)
        
        self._update_best(population, costs)
        self.convergence_history = [self.best_cost]
        
        for iteration in range(self.max_iterations):
            # Dynamic step size parameter (as in paper)
            w = self.w0 + 0.1 * (1.0 - iteration / self.max_iterations)
            
            new_population = []
            new_costs = []
            
            # Generate offspring
            for _ in range(self.population_size // 2):
                # Tournament selection for parents
                parent1 = self._tournament_selection(population, costs)
                parent2 = self._tournament_selection(population, costs)
                
                # Apply directional transport crossover
                offspring1, offspring2 = self._directional_transport_crossover(parent1, parent2, w)
                
                # Apply mutation
                offspring1 = self._polynomial_mutation(offspring1)
                offspring2 = self._polynomial_mutation(offspring2)
                
                new_population.extend([offspring1, offspring2])
                new_costs.extend([self.cost_function(offspring1), self.cost_function(offspring2)])
            
            # Replace population (elitism: keep best)
            combined_population = population + new_population
            combined_costs = costs + new_costs
            
            # Select best individuals for next generation
            sorted_indices = np.argsort(combined_costs)
            population = [combined_population[i] for i in sorted_indices[:self.population_size]]
            costs = [combined_costs[i] for i in sorted_indices[:self.population_size]]
            
            # FIXED: Always update best and record convergence, no early stopping
            previous_best = self.best_cost
            self._update_best(population, costs)
            
            # FIXED: Always record convergence history for ALL iterations
            self.convergence_history.append(self.best_cost)
            self._log_convergence(iteration, self.best_cost)
        
        return self.best_solution, self.best_cost, self.convergence_history
    
    def _tournament_selection(self, population, costs):
        """Select individual using tournament selection."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_costs = [costs[i] for i in indices]
        best_idx = indices[np.argmin(tournament_costs)]
        return population[best_idx]
    
    def _directional_transport_crossover(self, parent1, parent2, w):
        """Apply directional transport crossover as described in the paper."""
        try:
            # Calculate difference vector on manifold
            diff_vector = self._manifold_difference(parent1, parent2)
            
            # Generate random beta matrix/vector
            beta = self._generate_beta(parent1)
            
            # Create targets (pre-crossover)
            target1 = self._elementwise_multiply(beta, diff_vector)
            target1 = scale_tangent(target1, 0.5)  # ⊙ 1/2 from paper
            
            target2 = self._elementwise_multiply(beta, diff_vector)  
            target2 = scale_tangent(target2, -0.5)  # -beta ⊙ ... ⊙ 1/2 from paper
            
            # Apply directional transport operator
            candidate1 = self._apply_directional_transport(parent1, target1, w)
            candidate2 = self._apply_directional_transport(parent1, target2, w)
            
            return candidate1, candidate2
            
        except Exception as e:
            if self.verbose:
                print(f"Directional transport crossover error: {e}. Using parents.")
            return parent1, parent2
    
    def _manifold_difference(self, point1, point2):
        """Calculate difference between two points on manifold."""
        try:
            return self.manifold.log(point2, point1)  # point1 - point2
        except NotImplementedError:
            # Fallback for manifolds without logarithmic map
            if isinstance(point1, list) and isinstance(point2, list):
                return [p1 - p2 for p1, p2 in zip(point1, point2)]
            else:
                return point1 - point2
    
    def _generate_beta(self, point):
        """Generate random beta parameter for crossover."""
        if isinstance(point, list):
            return [np.random.rand(*p.shape) for p in point]
        else:
            return np.random.rand(*point.shape)
    
    def _elementwise_multiply(self, beta, vector):
        """Element-wise multiplication."""
        if isinstance(beta, list) and isinstance(vector, list):
            return [b * v for b, v in zip(beta, vector)]
        else:
            return beta * vector
    
    def _apply_directional_transport(self, x, z, w):
        """Apply directional transport operator."""
        try:
            # Project to tangent and normal spaces
            proj_tangent = self.manifold.projection(x, z)
            
            if isinstance(z, list):
                proj_normal = [z_i - p_i for z_i, p_i in zip(z, proj_tangent)]
            else:
                proj_normal = z - proj_tangent
            
            # Apply retraction with step size w
            movement_normal = scale_tangent(proj_normal, w)
            candidate = self.manifold.retraction(x, movement_normal)
            
            return candidate
            
        except Exception as e:
            if self.verbose:
                print(f"Directional transport error: {e}. Using original point.")
            return x
    
    def _polynomial_mutation(self, individual):
        """Apply polynomial mutation."""
        if np.random.rand() > self.mutation_rate:
            return individual
            
        try:
            if isinstance(individual, list):
                # For product manifolds, mutate each component
                mutated = []
                for elem in individual:
                    # Generate random tangent vector for mutation
                    mutation_vector = self.manifold.random_tangent_vector(elem)
                    mutation_strength = np.random.normal(0, 0.1)
                    scaled_mutation = scale_tangent(mutation_vector, mutation_strength)
                    mutated_elem = self.manifold.retraction(elem, scaled_mutation)
                    mutated.append(mutated_elem)
                return mutated
            else:
                # For single manifolds
                mutation_vector = self.manifold.random_tangent_vector(individual)
                mutation_strength = np.random.normal(0, 0.1)
                scaled_mutation = scale_tangent(mutation_vector, mutation_strength)
                return self.manifold.retraction(individual, scaled_mutation)
                
        except Exception as e:
            if self.verbose:
                print(f"Mutation error: {e}. Using original individual.")
            return individual
