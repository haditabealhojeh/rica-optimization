"""
Imperialist Competitive Algorithm (RICA) - RETRACTION-OPTIMIZED VERSION.
Uses retraction instead of logarithmic map for speed.
"""

import numpy as np
from typing import List, Tuple, Any
from .base_optimizer import BaseOptimizer
from manifolds.manifold_utils import scale_tangent, add_tangents, zero_vector


class RICAOptimizer(BaseOptimizer):
    """RICA using retraction-based operations for maximum speed."""
    
    def __init__(self, manifold, cost_function, population_size=50, max_iterations=100,
                 num_imperialists=None, assimilation_coef=0.7, revolution_rate=0.3,
                 assimilation_gamma=np.pi/6, seed=None, verbose=True):
        
        super().__init__(manifold, cost_function, population_size, max_iterations, seed, verbose)
        
        self.num_imperialists = num_imperialists or max(1, population_size // 8)
        self.assimilation_coef = assimilation_coef
        self.revolution_rate = revolution_rate
        self.assimilation_gamma = assimilation_gamma
        
    def _fast_approximate_log(self, x, y):
        """
        Approximate logarithmic map using retraction and projection.
        This is much faster than exact logarithmic map for most manifolds.
        """
        try:
            # Try exact logarithmic map first
            return self.manifold.log(x, y)
        except (NotImplementedError, AttributeError):
            # Fast approximation using projection of ambient difference
            if isinstance(x, list) and isinstance(y, list):
                # For product manifolds
                ambient_diff = [np.asarray(yi) - np.asarray(xi) for xi, yi in zip(x, y)]
                return self.manifold.projection(x, ambient_diff)
            else:
                # For single manifolds
                ambient_diff = np.asarray(y) - np.asarray(x)
                return self.manifold.projection(x, ambient_diff)
    
    def _fast_direction_estimation(self, col, imp):
        """
        Estimate direction from colony to imperialist using retraction-based approach.
        This avoids expensive logarithmic map computations.
        """
        # Method 1: Simple projection (fastest)
        try:
            if isinstance(col, list) and isinstance(imp, list):
                ambient_diff = [np.asarray(impi) - np.asarray(coli) for impi, coli in zip(imp, col)]
                return self.manifold.projection(col, ambient_diff)
            else:
                ambient_diff = np.asarray(imp) - np.asarray(col)
                return self.manifold.projection(col, ambient_diff)
        except:
            # Method 2: Random tangent vector scaled by simple distance measure
            random_dir = self.manifold.random_tangent_vector(col)
            # Simple distance approximation
            if isinstance(col, list) and isinstance(imp, list):
                dist_approx = np.sqrt(sum(np.sum((np.asarray(ci) - np.asarray(ii))**2) 
                                        for ci, ii in zip(col, imp)))
            else:
                dist_approx = np.linalg.norm(np.asarray(col) - np.asarray(imp))
            return scale_tangent(random_dir, dist_approx)
    
    def _fast_retraction_move(self, point, direction, step_size=1.0):
        """Fast retraction with step size control."""
        try:
            scaled_direction = scale_tangent(direction, step_size)
            return self.manifold.retraction(point, scaled_direction)
        except:
            # Fallback: simple addition for Euclidean-like spaces
            if isinstance(point, list) and isinstance(direction, list):
                return [pi + step_size * di for pi, di in zip(point, direction)]
            else:
                return point + step_size * direction

    def optimize(self) -> Tuple[Any, float, List[float]]:
        """Run RICA optimization using fast retraction-based operations."""
        
        # Initialize countries
        countries = [self.manifold.random_point() for _ in range(self.population_size)]
        country_costs = [self.cost_function(x) for x in countries]

        # Sort and select imperialists
        sorted_indices = np.argsort(country_costs)
        countries = [countries[i] for i in sorted_indices]
        country_costs = [country_costs[i] for i in sorted_indices]

        imperialists = countries[:self.num_imperialists]
        imperialist_costs = country_costs[:self.num_imperialists]
        colonies = countries[self.num_imperialists:]

        # Initialize empires
        empires = [[] for _ in range(len(imperialists))]
        for col in colonies:
            empires[np.random.randint(len(imperialists))].append(col)

        self.best_solution = imperialists[0]
        self.best_cost = imperialist_costs[0]
        self.convergence_history = [self.best_cost]
        
        # Pre-check manifold type
        is_product_manifold = hasattr(self.manifold, 'manifolds')

        for iteration in range(self.max_iterations):
            if not imperialists:
                if self.verbose:
                    print(f"Iteration {iteration+1}: No imperialists left. Stopping optimization.")
                break

            for i in range(len(imperialists)):
                imp = imperialists[i]
                current_empire_colonies = empires[i]
                
                if not current_empire_colonies:
                    continue
                    
                new_empire_colonies = []
                
                # Pre-compute costs for all colonies
                colony_costs_before = [self.cost_function(col) for col in current_empire_colonies]

                for j, col in enumerate(current_empire_colonies):
                    col_cost_before_move = colony_costs_before[j]

                    # FAST ASSIMILATION USING RETRACTION-BASED APPROACH
                    # Estimate direction to imperialist (much faster than logarithmic map)
                    v_direction = self._fast_direction_estimation(col, imp)
                    norm_v = self.manifold.norm(col, v_direction)

                    if norm_v < 1e-12:
                        # If direction is negligible, use small random move
                        tangent_movement = scale_tangent(
                            self.manifold.random_tangent_vector(col), 0.1)
                    else:
                        # ORTHOGONAL DEVIATION COMPONENT (original logic preserved)
                        rand_dir = self.manifold.random_tangent_vector(col)
                        inner_prod = self.manifold.inner_product(col, rand_dir, v_direction)
                        proj_val = inner_prod / (norm_v**2 + 1e-12)

                        # Compute orthogonal component
                        ortho_component = add_tangents(
                            scale_tangent(rand_dir, 1.0),
                            scale_tangent(v_direction, -proj_val)
                        )
                        
                        norm_ortho = self.manifold.norm(col, ortho_component)
                        if norm_ortho > 1e-12:
                            ortho_dir = scale_tangent(ortho_component, 1.0 / norm_ortho)
                        else:
                            ortho_dir = zero_vector(self.manifold, col)

                        # Combine main direction with orthogonal deviation
                        random_angle = np.random.uniform(-self.assimilation_gamma, self.assimilation_gamma)
                        deviation_scale = norm_v * np.tan(random_angle)

                        main_move = scale_tangent(v_direction, self.assimilation_coef)
                        ortho_move = scale_tangent(ortho_dir, deviation_scale)
                        tangent_movement = add_tangents(main_move, ortho_move)

                    # APPLY MOVEMENT USING RETRACTION
                    try:
                        new_col_candidate = self.manifold.retraction(col, tangent_movement)
                    except Exception as e:
                        if self.verbose and iteration % 200 == 0:
                            print(f"Retraction error: {e}. Using small random move.")
                        # Fallback: small random move
                        small_random = scale_tangent(
                            self.manifold.random_tangent_vector(col), 0.01)
                        new_col_candidate = self.manifold.retraction(col, small_random)

                    # REVOLUTION - preserved original logic
                    if np.random.rand() < self.revolution_rate:
                        revolution_vector = self.manifold.random_tangent_vector(new_col_candidate)
                        revolution_norm = self.manifold.norm(new_col_candidate, revolution_vector)
                        
                        if revolution_norm > 1e-12:
                            # Scale revolution by current movement magnitude
                            rev_scale = 0.1 * norm_v if norm_v > 1e-12 else 0.1
                            scaled_revolution = scale_tangent(revolution_vector, rev_scale / revolution_norm)
                            try:
                                new_col_candidate = self.manifold.retraction(new_col_candidate, scaled_revolution)
                            except Exception:
                                pass  # Keep candidate if revolution fails

                    # SELECTION - original logic preserved
                    new_col_cost = self.cost_function(new_col_candidate)
                    
                    if new_col_cost < col_cost_before_move:
                        new_empire_colonies.append(new_col_candidate)
                    else:
                        new_empire_colonies.append(col)

                empires[i] = new_empire_colonies

                # IMPERIALIST UPDATE - original logic preserved
                if empires[i]:
                    colony_costs = [self.cost_function(c) for c in empires[i]]
                    best_colony_idx = np.argmin(colony_costs)
                    best_colony_cost = colony_costs[best_colony_idx]

                    if best_colony_cost < imperialist_costs[i]:
                        # Swap imperialist with best colony
                        old_imperialist = imperialists[i]
                        imperialists[i] = empires[i][best_colony_idx]
                        imperialist_costs[i] = best_colony_cost

                        # Remove promoted colony and add old imperialist as colony
                        promoted_colony = empires[i][best_colony_idx]
                        empires[i] = [col for col in empires[i] if not self._points_equal(col, promoted_colony)]
                        empires[i].append(old_imperialist)

            # IMPERIALISTIC COMPETITION - less frequent for speed
            if iteration % 10 == 0 and len(imperialists) > 1:
                imperialists, imperialist_costs, empires = self._fast_imperialistic_competition(
                    imperialists, imperialist_costs, empires)

            # UPDATE GLOBAL BEST
            if imperialist_costs:
                current_best_idx = np.argmin(imperialist_costs)
                if imperialist_costs[current_best_idx] < self.best_cost:
                    self.best_solution = imperialists[current_best_idx]
                    self.best_cost = imperialist_costs[current_best_idx]

            self.convergence_history.append(self.best_cost)
            
            # Progress reporting
            if self.verbose and (iteration % 100 == 0 or iteration == self.max_iterations - 1):
                print(f"Iteration {iteration+1}/{self.max_iterations}, Best Cost: {self.best_cost:.6e}")

        # FINAL SOLUTION SELECTION
        all_entities_final = imperialists + [c for emp in empires for c in emp]
        if all_entities_final:
            all_costs_final = [self.cost_function(x) for x in all_entities_final]
            best_idx_final = np.argmin(all_costs_final)
            self.best_solution = all_entities_final[best_idx_final]
            self.best_cost = all_costs_final[best_idx_final]

        return self.best_solution, self.best_cost, self.convergence_history

    def _points_equal(self, point1, point2):
        """Efficient point comparison."""
        try:
            if hasattr(point1, 'shape') and hasattr(point2, 'shape'):
                return np.array_equal(point1, point2)
            elif isinstance(point1, list) and isinstance(point2, list):
                return all(np.array_equal(p1, p2) for p1, p2 in zip(point1, point2))
            return point1 == point2
        except:
            return False

    def _fast_imperialistic_competition(self, imperialists, imperialist_costs, empires):
        """Fast imperialistic competition."""
        if len(imperialists) <= 1:
            return imperialists, imperialist_costs, empires

        # Calculate empire powers
        empire_powers = []
        for i in range(len(imperialists)):
            if empires[i]:
                colony_costs = [self.cost_function(col) for col in empires[i]]
                total_power = imperialist_costs[i] + np.mean(colony_costs)
            else:
                total_power = imperialist_costs[i]
            empire_powers.append(total_power)

        # Find weakest empire
        weakest_idx = np.argmax(empire_powers)
        
        if empires[weakest_idx]:
            # Take weakest colony from weakest empire
            colony_costs = [self.cost_function(col) for col in empires[weakest_idx]]
            weakest_colony_idx = np.argmax(colony_costs)
            moving_colony = empires[weakest_idx].pop(weakest_colony_idx)
            
            # Give to strongest empire
            strongest_idx = np.argmin(empire_powers)
            empires[strongest_idx].append(moving_colony)

        return imperialists, imperialist_costs, empires