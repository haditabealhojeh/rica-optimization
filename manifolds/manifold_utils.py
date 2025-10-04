"""
Utilities for manifold operations and geometry.
"""

import numpy as np
from pymanopt.manifolds import (
    Sphere, Stiefel, Grassmann, Oblique, SpecialOrthogonalGroup,
    Product, Euclidean, SymmetricPositiveDefinite
)


def create_manifold(problem_type: str, n=None, m=None, p=None, d_ambient=None):
    """Create appropriate manifold for the given problem type."""
    
    if problem_type == "dominant":
        if n is None or p is None:
            raise ValueError("For 'dominant', 'n' (matrix dim) and 'p' (rank) must be provided.")
        return Grassmann(n, p)
    
    elif problem_type == "sdp":
        if n is None or p is None:
            raise ValueError("For 'sdp', 'n' (num_rows_Y) and 'p' (num_cols_Y) must be provided.")
        if p < 1:
            raise ValueError("For 'sdp', p (num_cols_Y) must be at least 1.")
        return Product([Sphere(p - 1)] * n)
    
    elif problem_type == "svd":
        if n is None or m is None or p is None:
            raise ValueError("For 'svd', 'n' (rows_A), 'm' (cols_A), and 'p' (rank) must be provided.")
        return Product([Grassmann(n, p), Grassmann(m, p)])
    
    elif problem_type == "procrustes":
        return Product([SpecialOrthogonalGroup(3), Euclidean(3, 10)])
    
    elif problem_type == "thomson":
        n_points = n
        if n_points is None or d_ambient is None:
            raise ValueError("For 'thomson', 'n' (n_points) and 'd_ambient' must be provided.")
        if d_ambient < 1:
            raise ValueError("For 'thomson', d_ambient must be at least 1.")
        return Product([Sphere(d_ambient - 1)] * n_points)
    
    elif problem_type == "stiffness":
        m_dim = n
        if m_dim is None or m_dim <= 0:
            raise ValueError("For 'stiffness', 'n' (dimension of K_P) must be a positive integer.")
        return SymmetricPositiveDefinite(m_dim)
    
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def scale_tangent(tangent_vector, scalar):
    """Scale a tangent vector by a scalar."""
    if isinstance(tangent_vector, list):
        return [scalar * np.asarray(tv_i) for tv_i in tangent_vector]
    return scalar * np.asarray(tangent_vector)


def add_tangents(tangent_vector1, tangent_vector2):
    """Add two tangent vectors."""
    if isinstance(tangent_vector1, list):
        if not isinstance(tangent_vector2, list) or len(tangent_vector1) != len(tangent_vector2):
            raise ValueError("For list addition, both tangent vectors must be lists of the same length.")
        return [np.asarray(tv1_i) + np.asarray(tv2_i) for tv1_i, tv2_i in zip(tangent_vector1, tangent_vector2)]
    return np.asarray(tangent_vector1) + np.asarray(tangent_vector2)


def zero_vector(manifold, point):
    """Return a zero tangent vector at the given point on the manifold."""
    
    def small_random_tangent(m, p):
        """Fallback: return a very small random tangent vector."""
        rand_tangent = m.random_tangent_vector(p)
        norm = m.norm(p, rand_tangent)
        return rand_tangent * (1e-6 / (norm + 1e-12))

    if isinstance(manifold, Product):
        return [zero_vector(m, p) for m, p in zip(manifold.manifolds, point)]
    elif isinstance(manifold, (Grassmann, Sphere, Oblique, Euclidean, SymmetricPositiveDefinite)):
        return np.zeros_like(point)
    elif isinstance(manifold, SpecialOrthogonalGroup):
        return small_random_tangent(manifold, point)
    else:
        return small_random_tangent(manifold, point)


def random_tangent_vector(manifold, point, scale=1.0):
    """Generate random tangent vector with optional scaling."""
    tangent = manifold.random_tangent_vector(point)
    if scale != 1.0:
        if isinstance(tangent, list):
            tangent = [scale * t for t in tangent]
        else:
            tangent = scale * tangent
    return tangent