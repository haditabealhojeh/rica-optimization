"""
Cost function definitions for different optimization problems.
"""

import numpy as np
from typing import Callable, Dict, Any


def define_cost_function(problem_type: str, A_matrix=None, other_data=None) -> Callable:
    """Define cost function for the given problem type."""
    
    if problem_type == "dominant":
        if A_matrix is None:
            raise ValueError("A_matrix needed for dominant cost.")
        
        def cost(X):
            return -0.5 * np.trace(X.T @ A_matrix @ X)
        return cost
    
    elif problem_type == "sdp":
        if A_matrix is None:
            raise ValueError("A_matrix needed for sdp cost.")
        
        def cost_sdp(Y_rows_list):
            Y_matrix = np.array(Y_rows_list)
            return np.trace(Y_matrix.T @ A_matrix @ Y_matrix)
        return cost_sdp
    
    elif problem_type == "svd":
        if A_matrix is None:
            raise ValueError("A_matrix needed for svd cost.")
        
        def cost_svd(UV_list):
            U, V = UV_list
            return -0.5 * np.linalg.norm(U.T @ A_matrix @ V, 'fro') ** 2
        return cost_svd
    
    elif problem_type == "procrustes":
        if other_data is None or "Atrue" not in other_data:
            raise ValueError("other_data with Atrue needed for procrustes cost.")
        
        A_true = other_data["Atrue"]
        
        def cost_procrustes(RA_list):
            R, A_optim = RA_list
            term = R @ A_optim
            return np.linalg.norm(term - A_true, 'fro') ** 2
        return cost_procrustes
    
    elif problem_type == "thomson":
        def cost_thomson(X_points_list):
            n_pts = len(X_points_list)
            if n_pts < 2:
                return 0.0
            
            total_energy = 0.0
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    p_i = np.asarray(X_points_list[i])
                    p_j = np.asarray(X_points_list[j])
                    diff_norm_sq = np.sum((p_i - p_j) ** 2)
                    total_energy += 1.0 / (diff_norm_sq + 1e-12)  # Avoid division by zero
            
            return total_energy
        return cost_thomson
    
    elif problem_type == "stiffness":
        if other_data is None or "X_stiff" not in other_data or "Y_stiff" not in other_data:
            raise ValueError("other_data with X_stiff and Y_stiff needed for stiffness cost.")
        
        X_s, Y_s = other_data["X_stiff"], other_data["Y_stiff"]
        
        def cost_stiffness(KP):
            return np.linalg.norm(X_s @ KP - Y_s, 'fro') ** 2
        return cost_stiffness
    
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def get_problem_description(problem_type: str) -> str:
    """Get description of the optimization problem."""
    descriptions = {
        "dominant": "Dominant Invariant Subspace: Find orthonormal basis of dominant subspace",
        "sdp": "Semidefinite Programming: Find positive semidefinite matrix with unit diagonals",
        "svd": "Truncated SVD: Find low-rank approximation of matrix",
        "procrustes": "Procrustes Problem: Find optimal rotation alignment",
        "thomson": "Thomson Problem: Find well-distributed points on sphere", 
        "stiffness": "Stiffness Learning: Learn stiffness matrix from demonstrations"
    }
    return descriptions.get(problem_type, "Unknown problem type")