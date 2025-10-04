"""
Manifolds package for optimization.
"""

from .manifold_utils import (
    create_manifold,
    scale_tangent,
    add_tangents,
    zero_vector
)

__all__ = [
    'create_manifold',
    'scale_tangent', 
    'add_tangents',
    'zero_vector'
]