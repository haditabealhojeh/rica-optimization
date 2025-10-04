"""
Problem data loading and preparation utilities.
"""

import os
import scipy.io
import numpy as np
from typing import Dict, Any, Optional, Tuple
from manifolds.manifold_utils import create_manifold


class ProblemLoader:
    """Load and prepare optimization problems from .mat files."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def load_problem(self, problem_name: str, problem_type: str, 
                    rank: Optional[int] = None, d_ambient: int = 3) -> Tuple[Any, Any, Dict]:
        """Load problem data and create manifold."""
        
        filepath = os.path.join(self.data_dir, f'{problem_name}.mat')
        
        try:
            mat_data = scipy.io.loadmat(filepath)
            if os.environ.get('VERBOSE', '0') == '1':
                print(f"Loaded variables from {problem_name}.mat: {list(mat_data.keys())}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Problem file not found: {filepath}")
        
        A_matrix = mat_data.get("A")
        other_data = {}
        n_dim, m_dim, p_dim = None, None, rank
        
        # Extract rank if available
        if p_dim is None and 'p' in mat_data:
            try:
                p_dim = int(mat_data['p'].item())
                if os.environ.get('VERBOSE', '0') == '1':
                    print(f"Rank/p_dim (p) loaded from file: {p_dim}")
            except (ValueError, AttributeError):
                p_dim = None
        
        # Problem-specific data extraction
        if problem_type == "dominant":
            if A_matrix is None:
                raise ValueError("Matrix 'A' not found for dominant problem.")
            n_dim = A_matrix.shape[0]
            if p_dim is None:
                raise ValueError("Rank 'p' needed for dominant problem.")
                
        elif problem_type == "sdp":
            if A_matrix is None:
                raise ValueError("Matrix 'A' not found for sdp problem.")
            n_dim = A_matrix.shape[0]
            if p_dim is None:
                raise ValueError("p_dim needed for sdp problem.")
                
        elif problem_type == "svd":
            if A_matrix is None:
                raise ValueError("Matrix 'A' not found for svd problem.")
            n_dim = A_matrix.shape[0]
            m_dim = A_matrix.shape[1]
            if p_dim is None:
                raise ValueError("Rank 'p' needed for svd problem.")
                
        elif problem_type == "procrustes":
            if "Atrue" not in mat_data:
                raise ValueError("'Atrue' not found for procrustes problem.")
            other_data["Atrue"] = mat_data["Atrue"].real
            
        elif problem_type == "thomson":
            if 'X_init' in mat_data and isinstance(mat_data['X_init'], np.ndarray) and mat_data['X_init'].ndim == 2:
                n_dim = mat_data['X_init'].shape[0]
            elif 'n_points' in mat_data:
                n_dim = int(mat_data['n_points'].item())
            else:
                n_dim = 20  # Default
            m_dim = d_ambient
            
        elif problem_type == "stiffness":
            if "X" not in mat_data or "Y" not in mat_data:
                raise ValueError("'X' and 'Y' matrices not found for stiffness problem.")
            other_data["X_stiff"] = mat_data["X"]
            other_data["Y_stiff"] = mat_data["Y"]
            n_dim = other_data["X_stiff"].shape[1]
            
        else:
            # Generic fallback
            if A_matrix is not None and A_matrix.ndim == 2:
                n_dim = A_matrix.shape[0]
                m_dim = A_matrix.shape[1]
        
        # Create manifold
        manifold_m_param = m_dim if problem_type != "thomson" else d_ambient
        manifold = create_manifold(
            problem_type, 
            n=n_dim, 
            m=manifold_m_param, 
            p=p_dim,
            d_ambient=d_ambient if problem_type == "thomson" else None
        )
        
        return manifold, A_matrix, other_data
    
    def get_available_problems(self) -> list:
        """Get list of available problem files."""
        if not os.path.exists(self.data_dir):
            return []
        
        mat_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mat')]
        return [os.path.splitext(f)[0] for f in mat_files]