"""
Geometry utilities for manifold operations.
"""

import numpy as np
from typing import Union, List
from pymanopt.manifolds import Product


def is_product_manifold(manifold) -> bool:
    """Check if manifold is a product manifold."""
    return isinstance(manifold, Product)


def get_manifold_dimension(manifold) -> int:
    """Get dimension of the manifold."""
    try:
        return manifold.dim
    except AttributeError:
        # For product manifolds, sum dimensions
        if is_product_manifold(manifold):
            return sum(m.dim for m in manifold.manifolds)
        return -1  # Unknown dimension


def manifold_distance(manifold, point1, point2) -> float:
    """Compute distance between two points on the manifold."""
    try:
        return manifold.dist(point1, point2)
    except (NotImplementedError, AttributeError):
        # Fallback: use Euclidean distance in ambient space
        if isinstance(point1, list) and isinstance(point2, list):
            # For product manifolds
            total_dist = 0.0
            for p1, p2 in zip(point1, point2):
                total_dist += np.linalg.norm(p1 - p2) ** 2
            return np.sqrt(total_dist)
        else:
            return np.linalg.norm(point1 - point2)


def random_tangent_vector(manifold, point, scale=1.0):
    """Generate random tangent vector with optional scaling."""
    tangent = manifold.random_tangent_vector(point)
    if scale != 1.0:
        if isinstance(tangent, list):
            tangent = [scale * t for t in tangent]
        else:
            tangent = scale * tangent
    return tangent


def project_to_tangent_space(manifold, point, vector):
    """Project vector to tangent space at point."""
    try:
        return manifold.projection(point, vector)
    except (NotImplementedError, AttributeError):
        # Simple fallback for common cases
        if hasattr(manifold, '_projection'):
            return manifold._projection(point, vector)
        return vector  # Return as-is if projection not available


def exponential_map(manifold, point, tangent_vector):
    """Apply exponential map if available, otherwise use retraction."""
    try:
        return manifold.exp(point, tangent_vector)
    except (NotImplementedError, AttributeError):
        return manifold.retraction(point, tangent_vector)


def logarithmic_map(manifold, point1, point2):
    """Apply logarithmic map if available."""
    try:
        return manifold.log(point1, point2)
    except (NotImplementedError, AttributeError):
        # Fallback approximation
        if isinstance(point1, list) and isinstance(point2, list):
            return [p2 - p1 for p1, p2 in zip(point1, point2)]
        else:
            return point2 - point1