"""
Manifold Particle Swarm Optimization (mPSO) implementation.
"""

import numpy as np
from typing import List, Tuple, Any
from .base_optimizer import BaseOptimizer
from manifolds.manifold_utils import scale_tangent, add_tangents, zero_vector


class MPSOOptimizer(BaseOptimizer):
    """Manifold Particle Swarm Optimization algorithm."""
    
    def __init__(self, manifold, cost_function, population_size=50, max_iterations=100,
                 w=0.729, c1=1.494, c2=1.494, seed=None, verbose=True):
        
        super().__init__(manifold, cost_function, population_size, max_iterations, seed, verbose)
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive weight
        self.c2 = c2  # social weight
        
    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run mPSO optimization."""
        
        # Initialize particles and velocities
        particles = self._initialize_population()
        velocities = [zero_vector(self.manifold, p) for p in particles]
        
        # Initialize personal best
        personal_best = particles.copy()
        personal_best_costs = self._evaluate_population(particles)
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_costs)
        self.best_solution = particles[global_best_idx]
        self.best_cost = personal_best_costs[global_best_idx]
        
        self.convergence_history = [self.best_cost]
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # Generate random factors
                r1, r2 = np.random.rand(), np.random.rand()
                
                try:
                    # Calculate direction to personal best
                    dir_personal = self.manifold.log(particles[i], personal_best[i])
                except NotImplementedError:
                    # Fallback for manifolds without logarithmic map
                    ambient_diff = np.asarray(personal_best[i]) - np.asarray(particles[i])
                    dir_personal = self.manifold.projection(particles[i], ambient_diff)
                
                try:
                    # Calculate direction to global best
                    dir_global = self.manifold.log(particles[i], self.best_solution)
                except NotImplementedError:
                    ambient_diff = np.asarray(self.best_solution) - np.asarray(particles[i])
                    dir_global = self.manifold.projection(particles[i], ambient_diff)
                
                # Update velocity
                inertia_term = scale_tangent(velocities[i], self.w)
                cognitive_term = scale_tangent(dir_personal, self.c1 * r1)
                social_term = scale_tangent(dir_global, self.c2 * r2)
                
                new_velocity = add_tangents(inertia_term, cognitive_term)
                new_velocity = add_tangents(new_velocity, social_term)
                
                # Update position
                try:
                    new_particle = self.manifold.retraction(particles[i], new_velocity)
                except Exception as e:
                    if self.verbose:
                        print(f"mPSO retraction error: {e}. Using random point.")
                    new_particle = self.manifold.random_point()
                    new_velocity = zero_vector(self.manifold, new_particle)
                
                particles[i] = new_particle
                velocities[i] = new_velocity
                
                # Update personal best
                current_cost = self.cost_function(particles[i])
                if current_cost < personal_best_costs[i]:
                    personal_best[i] = particles[i]
                    personal_best_costs[i] = current_cost
                    
                    # Update global best
                    if current_cost < self.best_cost:
                        self.best_solution = particles[i]
                        self.best_cost = current_cost
            
            self._log_convergence(iteration, self.best_cost)
            self.convergence_history.append(self.best_cost)
        
        return self.best_solution, self.best_cost, self.convergence_history