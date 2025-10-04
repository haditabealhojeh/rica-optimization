"""
Configuration settings for manifold optimization comparison project.
Paper-accurate parameters for mDTMA, keeping RICA and other existing configs.
"""

import numpy as np

class Config:
    # =========================================================================
    # GENERAL EXPERIMENT SETTINGS (From Paper Methodology)
    # =========================================================================
    SEED = 42
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    
    # Paper settings (from Section 4.1)
    DEFAULT_NUM_RUNS = 2           # 21 independent runs (using 2 for testing)
    DEFAULT_POPULATION_SIZE = 100   # pn = 100 (paper value)
    DEFAULT_ITERATIONS = 800        # maxit = 800 (paper value)
    
    # =========================================================================
    # ALGORITHM PARAMETERS (mDTMA EXACT PAPER VALUES, others kept from original)
    # =========================================================================
    
    # RICA parameters (Imperialist Competitive Algorithm) - KEEP ORIGINAL
    RICA_IMPERIALIST_RATIO = 0.1  # num_imperialists / num_countries
    RICA_ASSIMILATION_COEF = 0.1    # Standard parameter
    RICA_REVOLUTION_RATE = 0.7      # Standard parameter  
    RICA_ASSIMILATION_GAMMA = np.pi / 4   # Approximate paper value
    
    # mPSO parameters (Manifold Particle Swarm Optimization) - UPDATED FOR MATLAB COMPATIBILITY
    MPSO_NOSTALGIA = 1.4  # MATLAB parameter name (cognitive weight)
    MPSO_SOCIAL = 1.4     # MATLAB parameter name (social weight)
    # Note: Inertia weight is computed dynamically in MATLAB style
    
    # mDE parameters (Manifold Differential Evolution) - KEEP ORIGINAL
    MDE_MUTATION_FACTOR = 0.5
    MDE_CROSSOVER_RATE = 0.9
    
    # mSMANN parameters (Manifold Simulated Annealing) - KEEP ORIGINAL
    MSMANN_INITIAL_TEMPERATURE = 100.0
    MSMANN_COOLING_RATE = 0.95
    MSMANN_STEP_SIZE = 1.0
    
    # mDTMA parameters (Manifold Directional Transport - EXACT PAPER VALUES)
    MDTMA_STEP_SIZE = 0.1           # "w0 parameter" from Algorithm 1 line 4
    MDTMA_CROSSOVER_RATE = 0.8      # Standard GA parameter from paper
    MDTMA_MUTATION_RATE = 0.1       # Standard GA parameter from paper  
    MDTMA_TOURNAMENT_SIZE = 2       # "2-tournament preference" from Section 3.1
    
    # =========================================================================
    # PROBLEM-SPECIFIC SETTINGS (EXACT FROM PAPER TABLES 3-7)
    # =========================================================================
    
    # Thomson problem settings (Table 3)
    THOMSON_AMBIENT_DIM = 3         # "d = 3"
    THOMSON_POINTS = [50, 75, 100, 125, 150]  # "n = {50, 75, 100, 125, 150}"
    
    # Dominant invariant subspace (Table 5)
    DOMINANT_MATRIX_DIM = 128       # "The dimension of A is 128 × 128"
    DOMINANT_RANK = 3               # "p = 3"
    
    # SDP problem (Table 4)  
    SDP_MATRIX_DIM = 100            # "The dimension of A is 100 × 100"
    
    # Truncated SVD (Table 6)
    SVD_MATRIX_ROWS = 42            # "The dimension of A is set 42 × 60"
    SVD_MATRIX_COLS = 60
    SVD_RANK = 5                    # "p = 5"
    
    # General settings
    DEFAULT_RANK = 3
    PROCRUSTES_DIM = (3, 10, 50)
    # =========================================================================
    # VISUALIZATION SETTINGS - KEEP ORIGINAL
    # =========================================================================
    PLOT_STYLE = "default"
    FIGURE_SIZE = (12, 8)
    DPI = 600
    
    # Algorithm colors for consistent plotting - KEEP ORIGINAL
    ALGORITHM_COLORS = {
        'mPSO': '#E74C3C',          # Red
        'mDE': '#3498DB',           # Blue
        'mSMANN': '#2ECC71',        # Green
        'mDTMA': '#9B59B6',         # Purple
        'RICA': '#F39C12',          # Orange
        'steepest_descent': '#95A5A6',    # Gray
        'conjugate_gradient': '#34495E',  # Dark gray
        'trust_regions': '#1ABC9C'        # Teal
    }
    
    @classmethod
    def get_algorithm_params(cls, algorithm_name):
        """Get parameters for specific algorithm - mDTMA EXACT PAPER, others updated for MATLAB compatibility."""
        params = {
            'population_size': cls.DEFAULT_POPULATION_SIZE,
            'max_iterations': cls.DEFAULT_ITERATIONS
        }
        
        if algorithm_name == 'RICA':
            params.update({
                'num_imperialists': int(cls.DEFAULT_POPULATION_SIZE * cls.RICA_IMPERIALIST_RATIO),
                'assimilation_coef': cls.RICA_ASSIMILATION_COEF,
                'revolution_rate': cls.RICA_REVOLUTION_RATE,
                'assimilation_gamma': cls.RICA_ASSIMILATION_GAMMA
            })
        elif algorithm_name == 'mPSO':
            # UPDATED: Use MATLAB-compatible parameter names
            params.update({
                'nostalgia': cls.MPSO_NOSTALGIA,  # MATLAB parameter name
                'social': cls.MPSO_SOCIAL         # MATLAB parameter name
                # Note: inertia weight 'w' is computed dynamically in the optimizer
            })
        elif algorithm_name == 'mDE':
            params.update({
                'F': cls.MDE_MUTATION_FACTOR,
                'CR': cls.MDE_CROSSOVER_RATE
            })
        elif algorithm_name == 'mSMANN':
            params.update({
                'initial_temperature': cls.MSMANN_INITIAL_TEMPERATURE,
                'cooling_rate': cls.MSMANN_COOLING_RATE,
                'step_size': cls.MSMANN_STEP_SIZE
            })
        elif algorithm_name == 'mDTMA':
            params.update({
                'w0': cls.MDTMA_STEP_SIZE,  # Base value, overridden per problem
                'crossover_rate': cls.MDTMA_CROSSOVER_RATE,
                'mutation_rate': cls.MDTMA_MUTATION_RATE,
                'tournament_size': cls.MDTMA_TOURNAMENT_SIZE
            })
            
        return params
    
    @classmethod
    def get_problem_config(cls, problem_type):
        """Get configuration for specific problem type with PAPER-EXACT mDTMA parameters."""
        configs = {
            'thomson': {
                'd_ambient': cls.THOMSON_AMBIENT_DIM,
                'mdtma_w0': 0.1,  # EXACT PAPER: "set the w0 parameter at 0.1 for Thomson"
                'population_size': 100,  # EXACT PAPER: "100 population sizes"
                'max_iterations': 800    # EXACT PAPER: "800 iterations"
            },
            'dominant': {
                'n': cls.DOMINANT_MATRIX_DIM,
                'p': cls.DOMINANT_RANK,
                'mdtma_w0': 0.1,  # EXACT PAPER: "set the w0 parameter at 0.1 for Dominant"
                'population_size': 100,
                'max_iterations': 800
            },
            'sdp': {
                'n': cls.SDP_MATRIX_DIM,
                'mdtma_w0': 0.1,  # EXACT PAPER: "set the w0 parameter at 0.1 for ... Truncated SVD"
                'population_size': 100,
                'max_iterations': 800
            },
            'svd': {
                'n': cls.SVD_MATRIX_ROWS,
                'm': cls.SVD_MATRIX_COLS,
                'p': cls.SVD_RANK,
                'mdtma_w0': 0.1,  # EXACT PAPER: "set the w0 parameter at 0.1 for ... Truncated SVD"
                'population_size': 100,
                'max_iterations': 800
            },
            'procrustes': {
                'mdtma_w0': 0.4,  # EXACT PAPER: "The rest of the problems set the w0 parameter to 0.4"
                'population_size': 100,
                'max_iterations': 800
            },
            'stiffness': {
                'mdtma_w0': 0.4,  # EXACT PAPER: "The rest of the problems set the w0 parameter to 0.4"
                'population_size': 100, 
                'max_iterations': 800
            }
        }
        return configs.get(problem_type, {})
    
    @classmethod
    def get_paper_algorithms(cls):
        """Get list of algorithms used in paper comparisons."""
        return ['mPSO', 'mDE', 'mSMANN', 'mDTMA', 'RICA', 
                'steepest_descent', 'conjugate_gradient', 'trust_regions']
    
    @classmethod
    def get_metaheuristic_algorithms(cls):
        """Get only metaheuristic algorithms."""
        return ['mPSO', 'mDE', 'mSMANN', 'mDTMA', 'RICA']
    
    @classmethod
    def get_gradient_algorithms(cls):
        """Get only gradient-based algorithms."""
        return ['steepest_descent', 'conjugate_gradient', 'trust_regions']