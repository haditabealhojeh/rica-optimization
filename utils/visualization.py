
"""
Visualization utilities for optimization results - PLOT BEST RUN ONLY.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os

class ResultVisualizer:
    """Visualize optimization results and comparisons."""
    
    def __init__(self, results_dir: str = "results", style: str = "seaborn-v0_8"):
        self.results_dir = results_dir
        self.style = style
        os.makedirs(results_dir, exist_ok=True)

        # Set style
        plt.style.use(style)
        
        # Color scheme
        self.colors = {
            'mPSO': '#E74C3C',  # Red
            'mDE': '#3498DB',   # Blue
            'mSMANN': '#2ECC71', # Green
            'mDTMA': '#9B59B6', # Purple
            'RICA': '#F39C12',  # Orange
            'steepest_descent': '#95A5A6',  # Gray
            'conjugate_gradient': '#34495E', # Dark gray
            'trust_regions': '#1ABC9C'  # Teal
        }
    
    def _safe_plot_data(self, data):
        """Convert data to safe format for plotting."""
        if hasattr(data, 'tolist'):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [float(x) for x in data]
        else:
            return [float(data)]
    
    def _get_best_run_convergence(self, alg_results):
        """Get convergence history from the best run (lowest final cost)."""
        if not alg_results.get('costs') or not alg_results.get('convergence'):
            return None
        
        costs = [float(c) for c in alg_results['costs']]
        convergence_data = alg_results['convergence']
        
        # Find the run with the lowest final cost
        best_idx = np.argmin(costs)
        
        if best_idx < len(convergence_data):
            best_convergence = self._safe_plot_data(convergence_data[best_idx])
            return best_convergence
        return None
    
    def plot_convergence_comparison(self, results: Dict, problem_type: str, 
                              experiment_id: str = None,
                              save: bool = True, show: bool = True,
                              max_iterations: int = 800):
        """Plot convergence comparison using BEST RUN only."""
        
        # Generate filename with experiment_id
        if experiment_id:
            filename = f'{experiment_id}_convergence_comparison.png'
        else:
            filename = f'{problem_type}_convergence_comparison.png'
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Group algorithms
        metaheuristics = ['mPSO', 'mDE', 'mSMANN', 'mDTMA', 'RICA']
        gradient_based = ['steepest_descent', 'conjugate_gradient', 'trust_regions']
        
        # 1. Convergence curves - BEST RUN ONLY
        print(f"\nConvergence Comparison - Best Run Analysis:")
        print("-" * 50)
        
        for alg_name, alg_results in results.items():
            best_convergence = self._get_best_run_convergence(alg_results)
            
            if best_convergence is not None and len(best_convergence) > 0:
                # Plot the actual convergence history without padding
                iterations = range(len(best_convergence))
                
                # Plot style based on algorithm type
                linestyle = '-' if alg_name in metaheuristics else '--'
                linewidth = 2 if alg_name in metaheuristics else 1.5
                
                ax1.plot(iterations, best_convergence, label=alg_name, 
                        color=self.colors.get(alg_name, 'black'),
                        linestyle=linestyle, linewidth=linewidth)
                
                final_cost = best_convergence[-1] if best_convergence else float('inf')
                print(f"{alg_name:20} | Iterations: {len(best_convergence):4d} | Final Cost: {final_cost:.2e}")
                        
            else:
                print(f"{alg_name:20} | No convergence data available")
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title(f'Convergence Comparison (Best Run) - {problem_type.upper()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set x-axis limits to max_iterations
        ax1.set_xlim(0, max_iterations)
        
        # Safe log scale - only if all values are positive
        try:
            if ax1.get_lines():
                all_positive = True
                for line in ax1.get_lines():
                    y_data = line.get_ydata()
                    if len(y_data) > 0 and np.any(np.array(y_data) <= 0):
                        all_positive = False
                        break
                
                if all_positive:
                    ax1.set_yscale('log')
                    ax1.set_ylabel('Cost (log scale)')
        except:
            pass
        
        # 2. Final cost distribution (ALL RUNS - for statistical comparison)
        meta_costs, meta_labels = [], []
        grad_costs, grad_labels = [], []
        
        for alg_name, alg_results in results.items():
            if alg_results.get('costs'):
                costs = [float(c) for c in alg_results['costs']]
                if alg_name in metaheuristics:
                    meta_costs.append(costs)
                    meta_labels.append(alg_name)
                elif alg_name in gradient_based:
                    grad_costs.append(costs)
                    grad_labels.append(alg_name)
        
        # Create grouped box plot
        all_data = meta_costs + grad_costs
        all_labels = meta_labels + grad_labels
        
        if all_data:
            # Create positions with gap between groups
            positions = list(range(1, len(meta_costs) + 1)) + \
                       list(range(len(meta_costs) + 2, len(meta_costs) + len(grad_costs) + 2))
            
            box_plot = ax2.boxplot(all_data, positions=positions, labels=all_labels,
                                  patch_artist=True)
            
            # Color the boxes
            for i, (box, label) in enumerate(zip(box_plot['boxes'], all_labels)):
                box.set_facecolor(self.colors.get(label, 'white'))
                box.set_alpha(0.7)
            
            # Add separator line if we have both groups
            if meta_costs and grad_costs:
                ax2.axvline(x=len(meta_costs) + 0.5, color='red', linestyle='--', alpha=0.7)
                ax2.text(len(meta_costs) + 0.3, ax2.get_ylim()[1] * 0.9, 'Metaheuristics', 
                        rotation=90, fontsize=10)
                ax2.text(len(meta_costs) + 0.7, ax2.get_ylim()[1] * 0.9, 'Gradient-based', 
                        rotation=90, fontsize=10)
        
        ax2.set_ylabel('Final Cost')
        ax2.set_title('Final Cost Distribution (All Runs)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Runtime comparison
        runtimes, runtime_labels = [], []
        for alg_name, alg_results in results.items():
            if alg_results.get('runtimes'):
                runtimes.append([float(r) for r in alg_results['runtimes']])
                runtime_labels.append(alg_name)
        
        if runtimes:
            box_plot_runtime = ax3.boxplot(runtimes, labels=runtime_labels, patch_artist=True)
            # Color the boxes
            for i, (box, label) in enumerate(zip(box_plot_runtime['boxes'], runtime_labels)):
                box.set_facecolor(self.colors.get(label, 'white'))
                box.set_alpha(0.7)
            
            ax3.set_ylabel('Runtime (seconds)')
            ax3.set_title('Runtime Distribution (All Runs)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Performance summary table
        ax4.axis('off')
        table_data = []
        for alg_name in metaheuristics + gradient_based:
            if alg_name in results and results[alg_name].get('costs'):
                try:
                    costs = [float(c) for c in results[alg_name]['costs']]
                    runtimes = [float(r) for r in results[alg_name].get('runtimes', [0])]
                    
                    mean_cost = np.mean(costs)
                    std_cost = np.std(costs)
                    mean_runtime = np.mean(runtimes) if runtimes else 0
                    best_cost = min(costs) if costs else float('inf')
                    alg_type = 'Metaheuristic' if alg_name in metaheuristics else 'Gradient'
                    
                    table_data.append([alg_name, alg_type, 
                                     f'{best_cost:.2e}',  # Show best cost instead of mean
                                     f'{mean_runtime:.2f}s'])
                except Exception as e:
                    print(f"Warning: Could not process results for {alg_name}: {e}")
        
        if table_data:
            table = ax4.table(cellText=table_data,
                            colLabels=['Algorithm', 'Type', 'Best Cost', 'Mean Runtime'],
                            loc='center',
                            cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
        
        ax4.set_title('Performance Summary (Best Cost)')
        
        plt.tight_layout()
        
        if save:
            filename_path = os.path.join(self.results_dir, filename)
            plt.savefig(filename_path, dpi=600, bbox_inches='tight')
            print(f"✓ Saved convergence comparison plot to {filename_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_individual_convergence(self, results: Dict, problem_type: str,
                          experiment_id: str = None,
                          save: bool = True, show: bool = True,
                          max_iterations: int = 800):
        """Plot individual convergence curves for each algorithm - BEST RUN ONLY."""
        
        # Generate filename with experiment_id
        if experiment_id:
            filename = f'{experiment_id}_individual_convergence.png'
        else:
            filename = f'{problem_type}_individual_convergence.png'
        
        n_algorithms = len(results)
        if n_algorithms == 0:
            return
            
        # Create subplots
        n_cols = min(2, n_algorithms)
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle different cases for axes
        if n_algorithms == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)
        elif n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        print(f"\nIndividual Convergence - Best Run Analysis:")
        print("-" * 50)
        
        for idx, (alg_name, alg_results) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            best_convergence = self._get_best_run_convergence(alg_results)
            
            if best_convergence is not None and len(best_convergence) > 0:
                # Plot the best run convergence
                iterations = range(len(best_convergence))
                
                ax.plot(iterations, best_convergence, 
                       color=self.colors.get(alg_name, 'blue'), 
                       linewidth=2, label='Best Run')
                
                ax.set_title(f'{alg_name} (Best Run)')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Cost')
                ax.grid(True, alpha=0.3)
                
                # Set x-axis limits to actual iterations used
                ax.set_xlim(0, len(best_convergence))
                
                # Set log scale only if all values are positive
                if len(best_convergence) > 0 and np.all(np.array(best_convergence) > 0):
                    ax.set_yscale('log')
                    ax.set_ylabel('Cost (log scale)')
                
                # Add final cost annotation
                final_cost = best_convergence[-1]
                ax.text(0.05, 0.95, f'Final: {final_cost:.2e}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                print(f"{alg_name:20} | Iterations: {len(best_convergence):4d} | Final Cost: {final_cost:.2e}")
            else:
                ax.text(0.5, 0.5, 'No convergence data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{alg_name}')
        
        # Hide empty subplots
        for idx in range(len(results), len(axes)):
            try:
                axes[idx].set_visible(False)
            except:
                pass
        
        plt.suptitle(f'Individual Convergence (Best Run) - {problem_type.upper()}', fontsize=16)
        plt.tight_layout()
        
        if save:
            filename_path = os.path.join(self.results_dir, filename)
            plt.savefig(filename_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved individual convergence plot to {filename_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_runs_convergence(self, results: Dict, problem_type: str,
                          experiment_id: str = None,
                          save: bool = True, show: bool = True):
        """Plot ALL convergence runs for each algorithm (transparent lines)."""
        
        if experiment_id:
            filename = f'{experiment_id}_all_runs_convergence.png'
        else:
            filename = f'{problem_type}_all_runs_convergence.png'
        
        n_algorithms = len(results)
        if n_algorithms == 0:
            return
            
        # Create subplots
        n_cols = min(2, n_algorithms)
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle different cases for axes
        if n_algorithms == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)
        elif n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        for idx, (alg_name, alg_results) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            if alg_results.get('convergence') and len(alg_results['convergence']) > 0:
                convergence_data = alg_results['convergence']
                
                # Plot all runs as transparent lines
                for run_idx, conv in enumerate(convergence_data):
                    safe_conv = self._safe_plot_data(conv)
                    iterations = range(len(safe_conv))
                    
                    ax.plot(iterations, safe_conv, 
                           color=self.colors.get(alg_name, 'blue'), 
                           alpha=0.3, linewidth=1.0)
                
                # Highlight the best run
                best_convergence = self._get_best_run_convergence(alg_results)
                if best_convergence is not None:
                    iterations = range(len(best_convergence))
                    ax.plot(iterations, best_convergence,
                           color=self.colors.get(alg_name, 'red'),
                           linewidth=3, label='Best Run')
                
                ax.set_title(f'{alg_name}')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Cost')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set log scale if appropriate
                if len(best_convergence) > 0 and np.all(np.array(best_convergence) > 0):
                    ax.set_yscale('log')
                    ax.set_ylabel('Cost (log scale)')
        
        # Hide empty subplots
        for idx in range(len(results), len(axes)):
            try:
                axes[idx].set_visible(False)
            except:
                pass
        
        plt.suptitle(f'All Runs Convergence - {problem_type.upper()}', fontsize=16)
        plt.tight_layout()
        
        if save:
            filename_path = os.path.join(self.results_dir, filename)
            plt.savefig(filename_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved all runs convergence plot to {filename_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_thomson_sphere(self, results: Dict, problem_type: str, 
                      experiment_id: str = None,
                      n_points: int = 50, save: bool = True, show: bool = True):
        """Plot Thomson problem solutions on sphere - USING BEST RUN."""
        
        # Generate filename with experiment_id
        if experiment_id:
            filename = f'{experiment_id}_thomson_sphere.png'
        else:
            filename = f'{problem_type}_thomson_sphere.png'
        
        if problem_type != "thomson":
            print("Sphere plotting only available for Thomson problems")
            return
        
        n_algorithms = len(results)
        if n_algorithms == 0:
            return
        
        fig = plt.figure(figsize=(5 * min(n_algorithms, 4), 10))
        
        for idx, (alg_name, alg_results) in enumerate(results.items()):
            if not alg_results.get('costs'):
                continue
                
            # Get best solution (lowest final cost)
            costs = [float(c) for c in alg_results['costs']]
            best_idx = np.argmin(costs)
            
            if alg_results.get('convergence') and len(alg_results['convergence']) > best_idx:
                best_convergence = alg_results['convergence'][best_idx]
                # Get the actual solution from the best run
                best_solution = best_convergence[-1] if isinstance(best_convergence, list) and len(best_convergence) > 0 else None
            else:
                continue
                
            if best_solution is None:
                continue
            
            # Extract points (handle different solution formats)
            points = None
            if isinstance(best_solution, list):
                points = np.array(best_solution)
            elif hasattr(best_solution, 'shape'):
                points = best_solution
            elif isinstance(best_solution, np.ndarray):
                points = best_solution
            else:
                print(f"Warning: Cannot extract points for {alg_name}, skipping sphere plot")
                continue
            
            # Ensure points have correct shape
            if len(points.shape) == 1:
                points = points.reshape(-1, 3)  # Reshape to (n_points, 3)
            
            # Create 3D subplot
            ax = fig.add_subplot(2, min(n_algorithms, 4), idx + 1, projection='3d')
            
            # Plot points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      s=50, alpha=0.7, color=self.colors.get(alg_name, 'blue'))
            
            # Plot sphere surface
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5)
            
            best_cost = costs[best_idx]
            ax.set_title(f'{alg_name}\nCost: {best_cost:.2e}', fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set equal aspect ratio
            max_range = 1.2
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Set equal aspect ratio for 3D plot
            ax.set_box_aspect([1, 1, 1])
        
        plt.suptitle(f'Thomson Problem Solutions - Best Runs (n={n_points})', fontsize=16)
        plt.tight_layout()
        
        if save:
            filename_path = os.path.join(self.results_dir, filename)
            plt.savefig(filename_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Thomson sphere plot to {filename_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def save_results_to_file(results: Dict, problem_type: str, 
                        experiment_id: str, filename: str = None):
    """Save results to JSON file with experiment ID."""
    import json
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Generate filename if not provided
    if filename is None:
        import os
        from config import Config
        config = Config()
        filename = os.path.join(config.RESULTS_DIR, f'{experiment_id}_results.json')
    
    # Convert all data to native Python types for JSON serialization
    serializable_results = {}
    for alg_name, alg_data in results.items():
        serializable_results[alg_name] = {
            'costs': [float(c) for c in alg_data.get('costs', [])],
            'runtimes': [float(r) for r in alg_data.get('runtimes', [])],
            'convergence': [[float(val) for val in conv] for conv in alg_data.get('convergence', [])]
        }
    
    # Add experiment metadata
    experiment_data = {
        'experiment_id': experiment_id,
        'problem_type': problem_type,
        'timestamp': experiment_id.split('_')[-2] + '_' + experiment_id.split('_')[-1] if '_' in experiment_id else 'unknown',
        'results': serializable_results
    }
    
    with open(filename, 'w') as f:
        json.dump(experiment_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Results saved to {filename}")


def load_results_from_file(filename: str) -> Dict:
    """Load results from JSON file."""
    import json
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Return both metadata and results
        return {
            'metadata': {k: v for k, v in data.items() if k != 'results'},
            'results': data.get('results', {})
        }
    except Exception as e:
        print(f"Error loading results from {filename}: {e}")
        return {'metadata': {}, 'results': {}}