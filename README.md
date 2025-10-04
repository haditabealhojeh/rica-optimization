# Manifold Optimization Comparison

A comprehensive Python framework for comparing optimization algorithms on Riemannian manifolds, implementing the methods from "From constraints fusion to manifold optimization: A new directional transport manifold metaheuristic algorithm".

## Features

- **Multiple Algorithms**: 
  - mPSO (Manifold Particle Swarm Optimization)
  - mDE (Manifold Differential Evolution) 
  - mSMANN (Manifold Simulated Annealing)
  - mDTMA (Manifold Directional Transport Metaheuristic Algorithm)
  - RICA (Imperialist Competitive Algorithm)
  - Gradient-based methods (via Pymanopt)

- **Various Manifolds**:
  - Grassmann, Stiefel, Sphere manifolds
  - Symmetric Positive Definite matrices
  - Product manifolds
  - Special Orthogonal groups

- **Benchmark Problems**:
  - Dominant invariant subspace
  - Semidefinite programming
  - Truncated SVD
  - Procrustes problem
  - Thomson problem
  - Stiffness learning

## Installation

```bash
git clone https://github.com/your-username/manifold-optimization-comparison.git
cd manifold-optimization-comparison
pip install -r requirements.txt