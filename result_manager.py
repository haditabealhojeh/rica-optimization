"""
Utility for managing and analyzing multiple experiment results.
"""

import os
import json
import glob
from typing import Dict, List
import pandas as pd
from config import Config

class ResultsManager:
    """Manage and analyze multiple experiment results."""
    
    def __init__(self, results_dir: str = None):
        self.config = Config()
        self.results_dir = results_dir or self.config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
    
    def list_experiments(self, problem_type: str = None) -> List[Dict]:
        """List all experiments with their metadata."""
        pattern = os.path.join(self.results_dir, "*_results.json")
        experiment_files = glob.glob(pattern)
        
        experiments = []
        for file_path in experiment_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                experiment_info = {
                    'file': file_path,
                    'experiment_id': data.get('experiment_id', 'Unknown'),
                    'problem_type': data.get('problem_type', 'Unknown'),
                    'timestamp': data.get('timestamp', 'Unknown')
                }
                
                # Filter by problem type if specified
                if problem_type is None or experiment_info['problem_type'] == problem_type:
                    experiments.append(experiment_info)
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
    
    def load_experiment(self, experiment_id: str) -> Dict:
        """Load specific experiment by ID."""
        file_path = os.path.join(self.results_dir, f"{experiment_id}_results.json")
        if os.path.exists(file_path):
            return load_results_from_file(file_path)
        else:
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
    
    def compare_experiments(self, problem_type: str) -> pd.DataFrame:
        """Compare multiple experiments of the same problem type."""
        experiments = self.list_experiments(problem_type)
        
        comparison_data = []
        for exp in experiments:
            data = self.load_experiment(exp['experiment_id'])
            results = data['results']
            
            for alg_name, alg_data in results.items():
                if alg_data['costs']:
                    comparison_data.append({
                        'experiment_id': exp['experiment_id'],
                        'algorithm': alg_name,
                        'mean_cost': np.mean(alg_data['costs']),
                        'std_cost': np.std(alg_data['costs']),
                        'mean_runtime': np.mean(alg_data['runtimes']),
                        'best_cost': min(alg_data['costs']),
                        'timestamp': exp['timestamp']
                    })
        
        return pd.DataFrame(comparison_data)
    
    def delete_experiment(self, experiment_id: str):
        """Delete experiment files."""
        base_path = os.path.join(self.results_dir, experiment_id)
        files_to_delete = [
            f"{base_path}_results.json",
            f"{base_path}_convergence_comparison.png",
            f"{base_path}_individual_convergence.png",
            f"{base_path}_thomson_sphere.png"
        ]
        
        deleted_count = 0
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted: {file_path}")
        
        print(f"Deleted {deleted_count} files for experiment {experiment_id}")

# Usage example
if __name__ == "__main__":
    manager = ResultsManager()
    
    # List all dominant problem experiments
    dominant_experiments = manager.list_experiments("dominant")
    print("Dominant problem experiments:")
    for exp in dominant_experiments:
        print(f"  - {exp['experiment_id']} ({exp['timestamp']})")
    
    # Compare experiments
    comparison_df = manager.compare_experiments("dominant")
    print("\nComparison:")
    print(comparison_df.to_string())