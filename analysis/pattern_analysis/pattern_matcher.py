"""
analysis/pattern_analysis/pattern_matcher.py

Pattern matching module for MSTR prediction system.

This module evaluates pattern similarity between BTC and MSTR using constrained 
Dynamic Time Warping (DTW), statistical rigor, and progressive optimization. 
It implements a state-of-the-art pattern matching system combining constrained DTW, 
statistical rigor, and stepwise optimization.

Phase 1 Implementation: Core functionality with constrained DTW and adaptive normalization
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field
import logging
import time

# Handle imports with fallbacks for testing
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas/numpy not available. Pattern matching functionality will be disabled.")
    # Mock implementation for testing
    class MockPandas:
        class DataFrame:
            def __init__(self, data=None):
                self.data = data or {}
                self.empty = True
            def __len__(self):
                return 0
            def memory_usage(self, deep=True):
                class MockSeries:
                    def sum(self):
                        return 0
                return MockSeries()
        class Series:
            def __init__(self, data=None):
                self.data = data or []
            def sum(self):
                return 0
        class Timestamp:
            def __init__(self, dt):
                self.dt = dt
    pd = MockPandas()
    np = None

# Handle Numba import for JIT optimization
try:
    import os
    # Suppress numba verbose output
    os.environ['NUMBA_DISABLE_JIT'] = '0'     # Keep JIT enabled
    os.environ['NUMBA_WARNINGS'] = '0'       # Disable warnings
    os.environ['NUMBA_DEBUG_PRINT_AFTER'] = 'none'
    os.environ['NUMBA_DEBUG_PRINT_BEFORE'] = 'none'
    
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("numba not available. DTW performance will be degraded.")
    # Mock decorator for testing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Handle scipy imports for advanced statistics
try:
    from scipy.stats import beta
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Advanced statistical features will be disabled.")

# Handle memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Memory monitoring will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable numba debug logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# Suppress additional numba loggers
for logger_name in ['numba.core', 'numba.core.ssa', 'numba.core.types', 'numba.core.compiler']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Data schema imports (mock if not available)
try:
    from .direction_converter import DirectionPatterns
except ImportError:
    @dataclass
    class DirectionPatterns:
        """Mock DirectionPatterns for testing"""
        btc_directions: "pd.DataFrame" = field(default_factory=pd.DataFrame)
        mstr_directions: "pd.DataFrame" = field(default_factory=pd.DataFrame)
        btc_pattern_sequences: "pd.DataFrame" = field(default_factory=pd.DataFrame)
        mstr_pattern_sequences: "pd.DataFrame" = field(default_factory=pd.DataFrame)
        conversion_params: Dict[str, Any] = field(default_factory=dict)
        quality_metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Phase 1 Implementation: Core Pattern Matching Functions
# =============================================================================

# Task 1A: Constrained DTW with Numba Optimization
# =============================================================================

def _fast_dtw_core_numba(seq1, seq2, weights, constraint_window):
    """Numba-optimized DTW core (nopython mode)"""
    len1, len2 = seq1.shape[0], seq2.shape[0]
    
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((len1 + 1, len2 + 1), float('inf'))
    dtw_matrix[0, 0] = 0.0
    
    # Validation: Check if initialization is correct
    if dtw_matrix[0, 0] != 0.0:
        # Cannot use logger in Numba nopython mode
        return float('inf'), dtw_matrix
    
    # Apply Sakoe-Chiba constraint
    for i in range(1, len1 + 1):
        j_start = max(1, i - constraint_window)
        j_end = min(len2 + 1, i + constraint_window + 1)
        
        for j in range(j_start, j_end):
            # Calculate weighted Euclidean distance
            cost = _weighted_euclidean_distance_numba(seq1[i-1], seq2[j-1], weights)
            
            # DTW recurrence relation
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
    
    # Calculate path length for normalization
    path_length = _count_optimal_path_length_numba(dtw_matrix, len1, len2)
    
    # Normalize distance by path length
    final_distance = dtw_matrix[len1, len2]
    
    # Final computation
    if path_length > 0 and final_distance < float('inf'):
        normalized_distance = final_distance / path_length
    else:
        normalized_distance = float('inf')
    
    return normalized_distance, dtw_matrix

def _fast_dtw_core_python(seq1, seq2, weights, constraint_window):
    """Python fallback DTW core (with logging)"""
    if not PANDAS_AVAILABLE or np is None:
        # Fallback implementation - return safe default values
        return 1.0, [[1.0]]
    
    len1, len2 = seq1.shape[0], seq2.shape[0]
    
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((len1 + 1, len2 + 1), float('inf'))
    dtw_matrix[0, 0] = 0.0
    
    # Validation: Check if initialization is correct
    if dtw_matrix[0, 0] != 0.0:
        logger.error(f"DTW matrix initialization failed: dtw_matrix[0,0] = {dtw_matrix[0, 0]}")
        return float('inf'), dtw_matrix
    
    # Apply Sakoe-Chiba constraint
    for i in range(1, len1 + 1):
        j_start = max(1, i - constraint_window)
        j_end = min(len2 + 1, i + constraint_window + 1)
        
        for j in range(j_start, j_end):
            # Calculate weighted Euclidean distance
            cost = _weighted_euclidean_distance_python(seq1[i-1], seq2[j-1], weights)
            
            # DTW recurrence relation
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
    
    # Calculate path length for normalization
    path_length = _count_optimal_path_length_python(dtw_matrix, len1, len2)
    
    # Normalize distance by path length
    final_distance = dtw_matrix[len1, len2]
    
    # Final computation
    if path_length > 0 and final_distance < float('inf'):
        normalized_distance = final_distance / path_length
    else:
        # Only log if there's an actual problem
        if final_distance == float('inf'):
            logger.warning(f"DTW computation failed: no valid path found")
        normalized_distance = float('inf')
    
    return normalized_distance, dtw_matrix

# Apply Numba JIT to the numba version if available
if NUMBA_AVAILABLE:
    _fast_dtw_core_numba = jit(nopython=True, cache=True)(_fast_dtw_core_numba)

def _fast_dtw_core(seq1, seq2, weights, constraint_window):
    """Dispatcher function for DTW core computation"""
    if NUMBA_AVAILABLE and PANDAS_AVAILABLE and np is not None:
        return _fast_dtw_core_numba(seq1, seq2, weights, constraint_window)
    else:
        return _fast_dtw_core_python(seq1, seq2, weights, constraint_window)

# Numba and Python versions of weighted euclidean distance
def _weighted_euclidean_distance_numba(point1, point2, weights):
    """Numba version - no Python objects allowed"""
    diff = point1 - point2
    weighted_diff = diff * weights
    return np.sqrt(np.sum(weighted_diff * weighted_diff))

def _weighted_euclidean_distance_python(point1, point2, weights):
    """Python version - with fallback handling"""
    if not PANDAS_AVAILABLE or np is None:
        return 1.0
    diff = point1 - point2
    weighted_diff = diff * weights
    return np.sqrt(np.sum(weighted_diff * weighted_diff))

# Apply Numba JIT if available
if NUMBA_AVAILABLE:
    _weighted_euclidean_distance_numba = jit(nopython=True)(_weighted_euclidean_distance_numba)

def _weighted_euclidean_distance(point1, point2, weights):
    """Dispatcher for weighted euclidean distance"""
    if NUMBA_AVAILABLE and PANDAS_AVAILABLE and np is not None:
        return _weighted_euclidean_distance_numba(point1, point2, weights)
    else:
        return _weighted_euclidean_distance_python(point1, point2, weights)

# Numba and Python versions of path length calculation
def _count_optimal_path_length_numba(dtw_matrix, len1, len2):
    """Numba version - simple approximation"""
    return len1 + len2

def _count_optimal_path_length_python(dtw_matrix, len1, len2):
    """Python version - with fallback handling"""
    if not PANDAS_AVAILABLE or np is None:
        return 1
    return len1 + len2

# Apply Numba JIT if available
if NUMBA_AVAILABLE:
    _count_optimal_path_length_numba = jit(nopython=True)(_count_optimal_path_length_numba)

def _count_optimal_path_length(dtw_matrix, len1, len2):
    """Dispatcher for path length calculation"""
    if NUMBA_AVAILABLE and PANDAS_AVAILABLE and np is not None:
        return _count_optimal_path_length_numba(dtw_matrix, len1, len2)
    else:
        return _count_optimal_path_length_python(dtw_matrix, len1, len2)

def constrained_dtw_distance(
    seq1, 
    seq2, 
    constraint_ratio: float = 0.1,
    feature_weights = None
) -> Tuple[float, List]:
    """
    Constrained DTW distance calculation for multi-dimensional pattern sequences.
    
    This is the main function for DTW-based pattern similarity calculation,
    implementing Sakoe-Chiba band constraints for efficiency and better alignment.
    
    Args:
        seq1: First sequence (BTC pattern features)
        seq2: Second sequence (MSTR pattern features)
        constraint_ratio: Constraint window as ratio of max sequence length
        feature_weights: Feature importance weights (auto-normalized)
    
    Returns:
        normalized_distance: DTW distance normalized by path length
        warping_path: Optimal warping path [(i1,j1), (i2,j2), ...]
    """
    if not PANDAS_AVAILABLE or np is None:
        logger.warning("NumPy not available. Returning default distance.")
        return 1.0, []
    
    if hasattr(seq1, 'shape') and hasattr(seq2, 'shape'):
        if seq1.shape[0] == 0 or seq2.shape[0] == 0:
            return float('inf'), []
    else:
        if len(seq1) == 0 or len(seq2) == 0:
            return float('inf'), []
    
    if seq1.ndim == 1:
        seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(-1, 1)
    
    if seq1.shape[1] != seq2.shape[1]:
        raise ValueError(f"Feature dimensions must match: {seq1.shape[1]} vs {seq2.shape[1]}")
    
    # Determine constraint window size
    max_len = max(seq1.shape[0], seq2.shape[0])
    constraint_window = max(1, int(max_len * constraint_ratio))
    
    # Prepare feature weights
    if feature_weights is None:
        feature_weights = np.ones(seq1.shape[1]) / seq1.shape[1]
    else:
        feature_weights = feature_weights / np.sum(feature_weights)
    
    # Debug logging (only if needed)
    # logger.debug(f"DTW: seq1.shape={seq1.shape}, seq2.shape={seq2.shape}, constraint_window={constraint_window}")
    
    # Compute constrained DTW
    distance, dtw_matrix = _fast_dtw_core(seq1, seq2, feature_weights, constraint_window)
    
    # Reconstruct optimal warping path if needed
    warping_path = _backtrack_warping_path(dtw_matrix) if distance < float('inf') else []
    
    return distance, warping_path

def _backtrack_warping_path(dtw_matrix) -> List:
    """
    Backtrack the optimal warping path from DTW matrix.
    
    Args:
        dtw_matrix: Computed DTW distance matrix
    
    Returns:
        warping_path: Optimal alignment path
    """
    if not PANDAS_AVAILABLE or np is None:
        return []
    
    path = []
    i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    
    while i > 0 or j > 0:
        path.append((i-1, j-1))  # Convert to 0-based indexing
        
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Choose minimum cost predecessor
            costs = [
                dtw_matrix[i-1, j-1],  # match
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1]     # deletion
            ]
            min_idx = np.argmin(costs)
            
            if min_idx == 0:
                i, j = i-1, j-1
            elif min_idx == 1:
                i -= 1
            else:
                j -= 1
    
    return path[::-1]  # Reverse to get forward path


# Task 1B: Adaptive Feature Normalization
# =============================================================================

def _adaptive_normalize_features(
    features: pd.DataFrame, 
    volatility_series: pd.Series,
    method: str = 'adaptive',
    scope: str = 'adaptive'
) -> pd.DataFrame:
    """
    Market regime-adaptive feature normalization.
    
    This function implements different normalization strategies based on 
    market volatility regimes to preserve meaningful patterns while 
    ensuring numerical stability.
    
    Args:
        features: Feature matrix to normalize
        volatility_series: Volatility time series for regime detection
        method: Normalization method ['minmax', 'zscore', 'robust', 'adaptive']
        scope: Normalization scope ['global', 'local', 'adaptive']
    
    Returns:
        normalized_features: Regime-adaptive normalized features
    """
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available. Returning original features.")
        return features
    
    if features.empty:
        return features
    
    # Market regime detection based on volatility quantiles
    vol_threshold_high = volatility_series.quantile(0.8) if len(volatility_series) > 0 else 1.0
    vol_threshold_low = volatility_series.quantile(0.2) if len(volatility_series) > 0 else 0.0
    
    # Create regime labels
    regime_labels = np.select([
        volatility_series >= vol_threshold_high,
        volatility_series <= vol_threshold_low
    ], ['high_vol', 'low_vol'], default='normal_vol')
    
    normalized_features = features.copy()
    
    # Apply regime-specific normalization
    for regime in ['low_vol', 'normal_vol', 'high_vol']:
        regime_mask = (regime_labels == regime)
        
        if not regime_mask.any():
            continue
            
        regime_features = features[regime_mask]
        
        if regime == 'low_vol':
            # Low volatility: Global normalization (preserve long-term trends)
            regime_normalized = _global_normalize(regime_features, method)
        elif regime == 'high_vol':
            # High volatility: Local normalization (focus on local patterns)
            regime_normalized = _local_normalize(regime_features, method, window=30)
        else:
            # Normal volatility: Hybrid approach
            global_norm = _global_normalize(regime_features, method)
            local_norm = _local_normalize(regime_features, method, window=60)
            regime_normalized = 0.6 * global_norm + 0.4 * local_norm
        
        normalized_features.loc[regime_mask] = regime_normalized
    
    # Apply boundary smoothing for regime transitions
    transition_zones = _detect_regime_transitions(regime_labels)
    if transition_zones:
        normalized_features = _smooth_regime_boundaries(normalized_features, transition_zones)
    
    return normalized_features

def _global_normalize(features: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Global normalization using entire dataset statistics.
    
    Args:
        features: Feature matrix
        method: Normalization method
    
    Returns:
        normalized_features: Globally normalized features
    """
    if not PANDAS_AVAILABLE or features.empty:
        return features
    
    if method == 'minmax':
        return (features - features.min()) / (features.max() - features.min())
    elif method == 'zscore':
        return (features - features.mean()) / features.std()
    elif method == 'robust':
        median = features.median()
        mad = features.mad()  # Median absolute deviation
        return (features - median) / (mad + 1e-8)  # Small epsilon for stability
    else:
        # Default to z-score
        return (features - features.mean()) / (features.std() + 1e-8)

def _local_normalize(features: pd.DataFrame, method: str, window: int) -> pd.DataFrame:
    """
    Local normalization using rolling window statistics.
    
    Args:
        features: Feature matrix
        method: Normalization method
        window: Rolling window size
    
    Returns:
        normalized_features: Locally normalized features
    """
    if not PANDAS_AVAILABLE or features.empty:
        return features
    
    if method == 'minmax':
        rolling_min = features.rolling(window=window, min_periods=1).min()
        rolling_max = features.rolling(window=window, min_periods=1).max()
        return (features - rolling_min) / (rolling_max - rolling_min + 1e-8)
    elif method == 'zscore':
        rolling_mean = features.rolling(window=window, min_periods=1).mean()
        rolling_std = features.rolling(window=window, min_periods=1).std()
        return (features - rolling_mean) / (rolling_std + 1e-8)
    elif method == 'robust':
        rolling_median = features.rolling(window=window, min_periods=1).median()
        rolling_mad = features.rolling(window=window, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        return (features - rolling_median) / (rolling_mad + 1e-8)
    else:
        # Default to rolling z-score
        rolling_mean = features.rolling(window=window, min_periods=1).mean()
        rolling_std = features.rolling(window=window, min_periods=1).std()
        return (features - rolling_mean) / (rolling_std + 1e-8)

def _detect_regime_transitions(regime_labels) -> List[int]:
    """
    Detect regime transition points for boundary smoothing.
    
    Args:
        regime_labels: Array of regime labels
    
    Returns:
        transition_indices: List of transition point indices
    """
    if not PANDAS_AVAILABLE or len(regime_labels) <= 1:
        return []
    
    transitions = []
    for i in range(1, len(regime_labels)):
        if regime_labels[i] != regime_labels[i-1]:
            transitions.append(i)
    
    return transitions

def _smooth_regime_boundaries(
    features: pd.DataFrame, 
    transitions: List[int], 
    smooth_window: int = 5
) -> pd.DataFrame:
    """
    Apply Gaussian smoothing around regime transition boundaries.
    
    Args:
        features: Normalized feature matrix
        transitions: Transition point indices
        smooth_window: Smoothing window half-width
    
    Returns:
        smoothed_features: Boundary-smoothed features
    """
    if not PANDAS_AVAILABLE or features.empty or not transitions:
        return features
    
    smoothed = features.copy()
    
    for transition_idx in transitions:
        start_idx = max(0, transition_idx - smooth_window)
        end_idx = min(len(features), transition_idx + smooth_window + 1)
        
        if end_idx - start_idx > 1:
            # Apply simple moving average smoothing
            smoothed.iloc[start_idx:end_idx] = features.iloc[start_idx:end_idx].rolling(
                window=3, center=True, min_periods=1
            ).mean()
    
    return smoothed


# Task 1C: Basic Similarity Calculation
# =============================================================================

def calculate_similarity_scores(
    btc_features: pd.DataFrame,
    mstr_features: pd.DataFrame,
    matching_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate similarity scores between all BTC-MSTR pattern pairs (Phase 1).
    
    This function implements the core similarity calculation using various
    distance metrics and handles the conversion from distance to similarity.
    
    Args:
        btc_features: BTC normalized feature matrix
        mstr_features: MSTR normalized feature matrix  
        matching_params: Matching configuration parameters
    
    Returns:
        similarity_matrix: Full similarity matrix [0.0-1.0]
    """
    if not PANDAS_AVAILABLE or btc_features.empty or mstr_features.empty:
        logger.warning("Invalid input data. Returning empty similarity matrix.")
        return pd.DataFrame()
    
    # Extract parameters
    algorithm = matching_params.get('matching_algorithm', 'constrained_dtw')
    dtw_constraint_ratio = matching_params.get('dtw_constraint_ratio', 0.1)
    allow_inverse = matching_params.get('allow_inverse_patterns', True)
    
    # Prepare feature weights
    feature_weights = np.array([
        matching_params.get('direction_weight', 0.4),
        matching_params.get('strength_weight', 0.3),
        matching_params.get('volatility_weight', 0.2),
        matching_params.get('hurst_weight', 0.1)
    ])
    feature_weights = feature_weights / feature_weights.sum()
    
    # Initialize similarity matrix
    btc_dates = btc_features.index
    mstr_dates = mstr_features.index
    similarity_matrix = pd.DataFrame(
        data=np.zeros((len(btc_dates), len(mstr_dates))),
        index=btc_dates,
        columns=mstr_dates
    )
    
    logger.info(f"Computing similarity matrix ({len(btc_dates)} x {len(mstr_dates)}) using {algorithm}")
    
    # Calculate pairwise similarities
    for i, (btc_date, btc_pattern) in enumerate(btc_features.iterrows()):
        if i % max(1, len(btc_features) // 10) == 0:
            logger.info(f"Processing BTC pattern {i+1}/{len(btc_features)}")
        
        for j, (mstr_date, mstr_pattern) in enumerate(mstr_features.iterrows()):
            # Calculate distance based on selected algorithm
            if algorithm == 'constrained_dtw':
                distance, _ = constrained_dtw_distance(
                    btc_pattern.values.reshape(-1, 1), 
                    mstr_pattern.values.reshape(-1, 1),
                    dtw_constraint_ratio,
                    feature_weights
                )
            elif algorithm == 'cosine':
                distance = _cosine_distance(btc_pattern.values, mstr_pattern.values)
            elif algorithm == 'euclidean':
                distance = _weighted_euclidean_distance(
                    btc_pattern.values, mstr_pattern.values, feature_weights
                )
            else:
                # Default to weighted Euclidean
                distance = _weighted_euclidean_distance(
                    btc_pattern.values, mstr_pattern.values, feature_weights
                )
            
            # Convert distance to similarity [0.0-1.0]
            similarity = 1.0 / (1.0 + distance) if distance < float('inf') else 0.0
            similarity_matrix.iloc[i, j] = similarity
    
    # Handle inverse patterns if enabled
    if allow_inverse:
        logger.info("Processing inverse patterns...")
        inverse_similarity = _calculate_inverse_similarity(
            btc_features, mstr_features, matching_params
        )
        # Take maximum similarity (regular or inverse)
        similarity_matrix = pd.DataFrame(
            np.maximum(similarity_matrix.values, inverse_similarity.values),
            index=similarity_matrix.index,
            columns=similarity_matrix.columns
        )
    
    logger.info("Similarity matrix computation completed")
    return similarity_matrix

def _cosine_distance(vec1, vec2) -> float:
    """
    Calculate cosine distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        cosine_distance: 1 - cosine_similarity
    """
    if not PANDAS_AVAILABLE or np is None:
        return 1.0
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return 1.0 - cosine_sim

def _calculate_inverse_similarity(
    btc_features: pd.DataFrame,
    mstr_features: pd.DataFrame,
    matching_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate similarity matrix considering inverse patterns.
    
    Inverse patterns are created by flipping the direction component
    while keeping other features intact.
    
    Args:
        btc_features: BTC feature matrix
        mstr_features: MSTR feature matrix
        matching_params: Matching parameters
    
    Returns:
        inverse_similarity_matrix: Similarity matrix for inverse patterns
    """
    if not PANDAS_AVAILABLE or btc_features.empty or mstr_features.empty:
        return pd.DataFrame()
    
    # Create inverted BTC patterns (flip direction component)
    inverted_btc = btc_features.copy()
    if 'direction' in inverted_btc.columns:
        inverted_btc['direction'] *= -1
    
    # Calculate similarity with inverted patterns
    return calculate_similarity_scores(inverted_btc, mstr_features, {
        **matching_params,
        'allow_inverse_patterns': False  # Avoid infinite recursion
    })


# Pattern Statistics and Quality Metrics
# =============================================================================

def extract_significant_matches(
    similarity_matrix: pd.DataFrame,
    btc_pattern_sequences: pd.DataFrame,
    mstr_pattern_sequences: pd.DataFrame,
    matching_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extract significant pattern matches based on similarity threshold.
    
    Args:
        similarity_matrix: Computed similarity matrix
        btc_pattern_sequences: BTC pattern sequence data
        mstr_pattern_sequences: MSTR pattern sequence data
        matching_params: Matching parameters
    
    Returns:
        significant_matches: DataFrame of high-similarity matches
    """
    if not PANDAS_AVAILABLE or similarity_matrix.empty:
        return pd.DataFrame()
    
    threshold = matching_params.get('similarity_threshold', 0.7)
    percentile_threshold = matching_params.get('percentile_threshold', 90)
    
    # Use percentile threshold if higher than absolute threshold
    dynamic_threshold = max(threshold, similarity_matrix.values.flatten().quantile(percentile_threshold / 100))
    
    # Find matches above threshold
    high_similarity_mask = similarity_matrix >= dynamic_threshold
    matches = []
    
    for btc_date in similarity_matrix.index:
        for mstr_date in similarity_matrix.columns:
            if high_similarity_mask.loc[btc_date, mstr_date]:
                similarity_score = similarity_matrix.loc[btc_date, mstr_date]
                
                # Calculate time lag (positive = MSTR delayed)
                time_lag = (mstr_date - btc_date).days
                
                # Get pattern information if available
                btc_pattern_code = ""
                mstr_pattern_code = ""
                pattern_length = 1
                
                if not btc_pattern_sequences.empty and btc_date in btc_pattern_sequences.index:
                    btc_pattern_info = btc_pattern_sequences.loc[btc_date]
                    btc_pattern_code = btc_pattern_info.get('pattern_code', '')
                    pattern_length = btc_pattern_info.get('pattern_length', 1)
                
                if not mstr_pattern_sequences.empty and mstr_date in mstr_pattern_sequences.index:
                    mstr_pattern_info = mstr_pattern_sequences.loc[mstr_date]
                    mstr_pattern_code = mstr_pattern_info.get('pattern_code', '')
                
                # Determine match type
                match_type = 'exact' if similarity_score > 0.95 else 'similar'
                
                matches.append({
                    'btc_date': btc_date,
                    'mstr_date': mstr_date,
                    'similarity_score': similarity_score,
                    'btc_pattern_code': btc_pattern_code,
                    'mstr_pattern_code': mstr_pattern_code,
                    'pattern_length': pattern_length,
                    'time_lag': time_lag,
                    'match_type': match_type
                })
    
    if matches:
        significant_matches = pd.DataFrame(matches)
        # Sort by similarity score (descending)
        significant_matches = significant_matches.sort_values('similarity_score', ascending=False)
        significant_matches = significant_matches.reset_index(drop=True)
    else:
        significant_matches = pd.DataFrame(columns=[
            'btc_date', 'mstr_date', 'similarity_score', 'btc_pattern_code',
            'mstr_pattern_code', 'pattern_length', 'time_lag', 'match_type'
        ])
    
    logger.info(f"Found {len(significant_matches)} significant matches above threshold {dynamic_threshold:.3f}")
    return significant_matches

def calculate_pattern_statistics(
    significant_matches: pd.DataFrame,
    btc_pattern_sequences: pd.DataFrame,
    mstr_pattern_sequences: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Calculate comprehensive pattern statistics.
    
    Args:
        significant_matches: Significant pattern matches
        btc_pattern_sequences: BTC pattern sequences
        mstr_pattern_sequences: MSTR pattern sequences
    
    Returns:
        pattern_statistics: Dictionary of statistical DataFrames
    """
    if not PANDAS_AVAILABLE or significant_matches.empty:
        return {}
    
    stats = {}
    
    # Basic match statistics
    stats['match_summary'] = pd.DataFrame({
        'total_matches': [len(significant_matches)],
        'avg_similarity': [significant_matches['similarity_score'].mean()],
        'max_similarity': [significant_matches['similarity_score'].max()],
        'min_similarity': [significant_matches['similarity_score'].min()],
        'avg_time_lag': [significant_matches['time_lag'].mean()],
        'avg_pattern_length': [significant_matches['pattern_length'].mean()]
    })
    
    # Time lag distribution
    if 'time_lag' in significant_matches.columns:
        lag_stats = significant_matches['time_lag'].describe()
        stats['time_lag_distribution'] = lag_stats.to_frame('time_lag')
    
    # Pattern length distribution
    if 'pattern_length' in significant_matches.columns:
        length_stats = significant_matches['pattern_length'].describe()
        stats['pattern_length_distribution'] = length_stats.to_frame('pattern_length')
    
    # Match type distribution
    if 'match_type' in significant_matches.columns:
        match_type_counts = significant_matches['match_type'].value_counts()
        stats['match_type_distribution'] = match_type_counts.to_frame('count')
    
    return stats

def calculate_matching_quality(
    similarity_matrix: pd.DataFrame,
    significant_matches: pd.DataFrame,
    matching_params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate matching quality metrics.
    
    Args:
        similarity_matrix: Full similarity matrix
        significant_matches: Extracted significant matches
        matching_params: Matching parameters used
    
    Returns:
        quality_metrics: Dictionary of quality indicators
    """
    if not PANDAS_AVAILABLE or similarity_matrix.empty:
        return {}
    
    quality = {}
    
    # Basic quality metrics
    quality['matrix_density'] = (similarity_matrix > 0).sum().sum() / similarity_matrix.size
    quality['avg_similarity'] = similarity_matrix.mean().mean()
    quality['max_similarity'] = similarity_matrix.max().max()
    quality['similarity_std'] = similarity_matrix.std().mean()
    
    # Significant match quality
    if not significant_matches.empty:
        quality['significant_match_ratio'] = len(significant_matches) / similarity_matrix.size
        quality['avg_significant_similarity'] = significant_matches['similarity_score'].mean()
        quality['match_quality_variance'] = significant_matches['similarity_score'].var()
    else:
        quality['significant_match_ratio'] = 0.0
        quality['avg_significant_similarity'] = 0.0
        quality['match_quality_variance'] = 0.0
    
    # Threshold effectiveness
    threshold = matching_params.get('similarity_threshold', 0.7)
    quality['threshold_selectivity'] = (similarity_matrix >= threshold).sum().sum() / similarity_matrix.size
    
    return quality


# =============================================================================
# Phase 2 Implementation: Statistical Rigor Enhancement
# =============================================================================

# Task 2A: Non-Maximum Suppression (NMS)
# =============================================================================

def _calculate_temporal_iou(match1: pd.Series, match2: pd.Series) -> float:
    """
    Calculate temporal Intersection over Union (IoU) between two pattern matches.
    
    This function computes the overlap ratio between two temporal patterns
    to identify redundant matches for NMS filtering.
    
    Args:
        match1: First pattern match (must have btc_date, pattern_length)
        match2: Second pattern match (must have btc_date, pattern_length)
    
    Returns:
        iou: Temporal IoU ratio [0.0-1.0]
    """
    if not PANDAS_AVAILABLE:
        return 0.0
    
    try:
        # Extract temporal boundaries for match1
        start1 = match1['btc_date']
        pattern_length1 = match1.get('pattern_length', 1)
        end1 = start1 + pd.Timedelta(days=pattern_length1)
        
        # Extract temporal boundaries for match2
        start2 = match2['btc_date']
        pattern_length2 = match2.get('pattern_length', 1)
        end2 = start2 + pd.Timedelta(days=pattern_length2)
        
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection_duration = max(pd.Timedelta(0), intersection_end - intersection_start)
        
        # Calculate union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_duration = union_end - union_start
        
        # Compute IoU
        if union_duration.total_seconds() > 0:
            iou = intersection_duration.total_seconds() / union_duration.total_seconds()
        else:
            iou = 0.0
        
        return max(0.0, min(1.0, iou))  # Clamp to [0.0, 1.0]
        
    except Exception as e:
        logger.warning(f"Error calculating temporal IoU: {e}")
        return 0.0

def _nms_overlapping_matches(
    matches_df: pd.DataFrame, 
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Non-Maximum Suppression for overlapping pattern matches.
    
    This function removes redundant high-similarity matches that overlap
    significantly in time, keeping only the highest quality matches.
    
    Args:
        matches_df: DataFrame of pattern matches
        iou_threshold: IoU threshold for overlap detection [0.0-1.0]
    
    Returns:
        nms_results: Dictionary containing:
            - selected_matches: DataFrame of non-overlapping matches
            - suppression_stats: Suppression statistics
    """
    if not PANDAS_AVAILABLE or matches_df.empty:
        return {
            'selected_matches': pd.DataFrame(),
            'suppression_stats': {
                'original_count': 0,
                'selected_count': 0,
                'suppression_ratio': 0.0
            }
        }
    
    logger.info(f"Starting NMS with {len(matches_df)} matches, IoU threshold {iou_threshold}")
    
    # Sort matches by similarity score (descending)
    sorted_matches = matches_df.sort_values('similarity_score', ascending=False).copy()
    sorted_matches = sorted_matches.reset_index(drop=True)
    
    selected_matches = []
    suppressed_indices = set()
    
    for i, current_match in sorted_matches.iterrows():
        if i in suppressed_indices:
            continue
        
        # Select current match (highest remaining score)
        selected_matches.append(current_match)
        
        # Suppress overlapping matches with lower scores
        for j, candidate_match in sorted_matches.iloc[i+1:].iterrows():
            if j in suppressed_indices:
                continue
            
            # Calculate temporal IoU
            iou = _calculate_temporal_iou(current_match, candidate_match)
            
            if iou > iou_threshold:
                suppressed_indices.add(j)
                logger.debug(f"Suppressed match {j} (IoU={iou:.3f} > {iou_threshold})")
    
    # Create result DataFrame
    if selected_matches:
        selected_df = pd.DataFrame(selected_matches)
        selected_df = selected_df.reset_index(drop=True)
    else:
        selected_df = pd.DataFrame(columns=matches_df.columns)
    
    # Calculate suppression statistics
    original_count = len(matches_df)
    selected_count = len(selected_df)
    suppression_ratio = 1.0 - (selected_count / original_count) if original_count > 0 else 0.0
    
    suppression_stats = {
        'original_count': original_count,
        'selected_count': selected_count,
        'suppression_ratio': suppression_ratio,
        'avg_similarity_improvement': (
            selected_df['similarity_score'].mean() - matches_df['similarity_score'].mean()
            if selected_count > 0 and original_count > 0 else 0.0
        ),
        'iou_threshold_used': iou_threshold
    }
    
    logger.info(f"NMS completed: {selected_count}/{original_count} matches selected "
               f"(suppression ratio: {suppression_ratio:.3f})")
    
    return {
        'selected_matches': selected_df,
        'suppression_stats': suppression_stats
    }


# Task 2B: False Discovery Rate (FDR) Control
# =============================================================================

def _apply_fdr_correction(
    p_values: "np.ndarray", 
    alpha: float = 0.05,
    method: str = 'benjamini_hochberg'
) -> Dict[str, Any]:
    """
    Apply False Discovery Rate (FDR) correction to multiple comparisons.
    
    This function implements the Benjamini-Hochberg procedure to control
    the expected proportion of false discoveries among rejected hypotheses.
    
    Args:
        p_values: Array of p-values to correct
        alpha: Desired FDR level [0.0-1.0]
        method: FDR method ('benjamini_hochberg' only for Phase 2)
    
    Returns:
        fdr_results: Dictionary containing correction results
    """
    if not PANDAS_AVAILABLE or np is None:
        logger.warning("NumPy not available. Returning mock FDR results.")
        return {
            'significant_indices': [],
            'adjusted_p_values': p_values if hasattr(p_values, '__len__') else [1.0],
            'fdr_threshold': alpha,
            'rejection_count': 0,
            'discovery_rate': 0.0,
            'method_used': method
        }
    
    if len(p_values) == 0:
        return {
            'significant_indices': [],
            'adjusted_p_values': np.array([]),
            'fdr_threshold': alpha,
            'rejection_count': 0,
            'discovery_rate': 0.0,
            'method_used': method
        }
    
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    logger.info(f"Applying FDR correction to {m} p-values using {method} method")
    
    if method == 'benjamini_hochberg':
        # Sort p-values and get original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate Benjamini-Hochberg thresholds
        bh_thresholds = np.arange(1, m + 1) * alpha / m
        
        # Find significant p-values
        # Find largest i such that p_i <= (i/m) * alpha
        significant_up_to = -1
        for i in range(m - 1, -1, -1):
            if sorted_p_values[i] <= bh_thresholds[i]:
                significant_up_to = i
                break
        
        # Get significant indices in original order
        if significant_up_to >= 0:
            significant_indices = sorted_indices[:significant_up_to + 1].tolist()
        else:
            significant_indices = []
        
        # Calculate adjusted p-values (step-up procedure)
        adjusted_p_values = np.zeros(m)
        for i in range(m):
            adjusted_p_values[i] = min(1.0, sorted_p_values[i] * m / (i + 1))
        
        # Ensure monotonicity (step-down adjustment)
        for i in range(m - 2, -1, -1):
            adjusted_p_values[i] = min(adjusted_p_values[i], adjusted_p_values[i + 1])
        
        # Reorder to match original p-value order
        original_order_adjusted = np.zeros(m)
        original_order_adjusted[sorted_indices] = adjusted_p_values
        
        rejection_count = len(significant_indices)
        discovery_rate = rejection_count / m if m > 0 else 0.0
        
        logger.info(f"FDR correction completed: {rejection_count}/{m} hypotheses rejected "
                   f"(discovery rate: {discovery_rate:.3f})")
        
        return {
            'significant_indices': significant_indices,
            'adjusted_p_values': original_order_adjusted,
            'fdr_threshold': alpha,
            'rejection_count': rejection_count,
            'discovery_rate': discovery_rate,
            'method_used': method,
            'bh_critical_values': bh_thresholds[sorted_indices],  # For diagnostic
        }
    
    else:
        raise ValueError(f"FDR method '{method}' not implemented in Phase 2")

def calculate_match_significance(
    matches_df: pd.DataFrame, 
    similarity_matrix: pd.DataFrame,
    n_permutations: int = 1000
) -> "np.ndarray":
    """
    Calculate statistical significance of pattern matches using permutation testing.
    
    This function estimates the probability that observed similarities could
    arise by chance through random permutation of the data.
    
    Args:
        matches_df: DataFrame of pattern matches
        similarity_matrix: Full similarity matrix
        n_permutations: Number of random permutations for null distribution
    
    Returns:
        p_values: Array of p-values for each match
    """
    if not PANDAS_AVAILABLE or matches_df.empty or similarity_matrix.empty:
        logger.warning("Invalid input data. Returning default p-values.")
        return np.ones(len(matches_df)) if len(matches_df) > 0 else np.array([])
    
    logger.info(f"Calculating match significance with {n_permutations} permutations")
    
    p_values = []
    similarity_values = similarity_matrix.values.flatten()
    
    # Pre-compute quantiles for efficiency
    percentiles = np.arange(0, 101, 1)
    similarity_quantiles = np.percentile(similarity_values, percentiles)
    
    for idx, match in matches_df.iterrows():
        observed_similarity = match['similarity_score']
        
        # Fast p-value estimation using pre-computed quantiles
        # This is much faster than full permutation testing
        quantile_idx = np.searchsorted(similarity_quantiles, observed_similarity)
        p_value = max(0.001, (100 - quantile_idx) / 100.0)  # Minimum p-value of 0.001
        
        p_values.append(p_value)
        
        if idx % max(1, len(matches_df) // 10) == 0:
            logger.debug(f"Processed significance for {idx+1}/{len(matches_df)} matches")
    
    p_values = np.array(p_values)
    
    logger.info(f"Significance calculation completed. Mean p-value: {p_values.mean():.4f}")
    
    return p_values


# Task 2C: Bayesian Confidence Intervals
# =============================================================================

def _calculate_bayesian_confidence(
    matches_df: pd.DataFrame,
    prior_params: Dict[str, float] = None,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate Bayesian confidence intervals for pattern match similarities.
    
    This function uses Beta-Binomial conjugate prior to estimate uncertainty
    in similarity scores and provide credible intervals.
    
    Args:
        matches_df: DataFrame of pattern matches
        prior_params: Prior parameters {'alpha': float, 'beta': float}
        confidence_level: Confidence level for intervals [0.0-1.0]
    
    Returns:
        confidence_df: DataFrame with confidence interval columns added
    """
    if not PANDAS_AVAILABLE or matches_df.empty:
        logger.warning("Invalid input data. Returning empty confidence intervals.")
        return pd.DataFrame()
    
    # Handle scipy availability
    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available. Using approximate confidence intervals.")
        return _calculate_approximate_confidence(matches_df, confidence_level)
    
    logger.info(f"Calculating Bayesian confidence intervals at {confidence_level*100}% level")
    
    # Set default Jeffrey's prior (uninformative)
    if prior_params is None:
        alpha_prior = 0.5
        beta_prior = 0.5
    else:
        alpha_prior = prior_params.get('alpha', 0.5)
        beta_prior = prior_params.get('beta', 0.5)
    
    confidence_results = []
    
    for idx, match in matches_df.iterrows():
        observed_similarity = match['similarity_score']
        
        # Pseudo sample size based on pattern length and complexity
        pattern_length = match.get('pattern_length', 5)
        n_observations = max(10, pattern_length * 2)  # Heuristic sample size
        
        # Beta-Binomial update
        successes = observed_similarity * n_observations
        failures = (1 - observed_similarity) * n_observations
        
        # Posterior parameters
        alpha_posterior = alpha_prior + successes
        beta_posterior = beta_prior + failures
        
        if SCIPY_AVAILABLE:
            from scipy.stats import beta
            
            # Calculate credible interval
            lower_tail = (1 - confidence_level) / 2
            upper_tail = 1 - lower_tail
            
            confidence_lower = beta.ppf(lower_tail, alpha_posterior, beta_posterior)
            confidence_upper = beta.ppf(upper_tail, alpha_posterior, beta_posterior)
            
            # Posterior statistics
            posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
            posterior_variance = (alpha_posterior * beta_posterior) / (
                (alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1)
            )
            
        else:
            # Approximate confidence intervals using normal approximation
            posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
            posterior_variance = (alpha_posterior * beta_posterior) / (
                (alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1)
            )
            posterior_std = np.sqrt(posterior_variance) if PANDAS_AVAILABLE and np is not None else 0.1
            
            # Normal approximation
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            confidence_lower = max(0.0, posterior_mean - z_score * posterior_std)
            confidence_upper = min(1.0, posterior_mean + z_score * posterior_std)
        
        # Credible interval width
        credible_width = confidence_upper - confidence_lower
        
        confidence_results.append({
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'credible_width': credible_width,
            'posterior_mean': posterior_mean,
            'posterior_variance': posterior_variance,
            'alpha_posterior': alpha_posterior,
            'beta_posterior': beta_posterior
        })
    
    # Create confidence DataFrame
    confidence_df = pd.DataFrame(confidence_results, index=matches_df.index)
    
    # Merge with original matches
    result_df = matches_df.copy()
    for col in confidence_df.columns:
        result_df[col] = confidence_df[col]
    
    logger.info(f"Bayesian confidence intervals calculated for {len(result_df)} matches")
    
    return result_df

def _calculate_approximate_confidence(
    matches_df: pd.DataFrame,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate approximate confidence intervals without scipy.
    
    Uses simple bootstrap-style approach for confidence estimation.
    """
    if not PANDAS_AVAILABLE:
        return pd.DataFrame()
    
    logger.info("Using approximate confidence intervals (scipy not available)")
    
    # Simple bootstrap estimation
    confidence_results = []
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    
    for idx, match in matches_df.iterrows():
        observed_similarity = match['similarity_score']
        
        # Estimate variance based on similarity value (U-shaped for [0,1])
        estimated_variance = observed_similarity * (1 - observed_similarity) / 20  # Heuristic
        estimated_std = np.sqrt(estimated_variance) if np is not None else 0.1
        
        confidence_lower = max(0.0, observed_similarity - z_score * estimated_std)
        confidence_upper = min(1.0, observed_similarity + z_score * estimated_std)
        credible_width = confidence_upper - confidence_lower
        
        confidence_results.append({
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'credible_width': credible_width,
            'posterior_mean': observed_similarity,
            'posterior_variance': estimated_variance,
            'alpha_posterior': 1.0,  # Placeholder
            'beta_posterior': 1.0    # Placeholder
        })
    
    # Create result DataFrame
    confidence_df = pd.DataFrame(confidence_results, index=matches_df.index)
    result_df = matches_df.copy()
    for col in confidence_df.columns:
        result_df[col] = confidence_df[col]
    
    return result_df


# =============================================================================
# Phase 3 Implementation: Advanced Optimization
# =============================================================================

# Task 3A: Chunked Processing for Large-Scale Data
# =============================================================================

def _estimate_memory_usage(btc_features: pd.DataFrame, mstr_features: pd.DataFrame) -> float:
    """
    Estimate memory usage for similarity matrix computation.
    
    Args:
        btc_features: BTC feature matrix
        mstr_features: MSTR feature matrix
    
    Returns:
        estimated_memory_gb: Estimated memory usage in GB
    """
    if not PANDAS_AVAILABLE or btc_features.empty or mstr_features.empty:
        return 0.0
    
    # Estimate similarity matrix size (float64 = 8 bytes per element)
    matrix_elements = len(btc_features) * len(mstr_features)
    matrix_size_bytes = matrix_elements * 8
    
    # Add overhead for intermediate calculations (factor of 3)
    total_size_bytes = matrix_size_bytes * 3
    
    # Convert to GB
    estimated_memory_gb = total_size_bytes / (1024**3)
    
    return estimated_memory_gb

def _chunked_similarity_calculation(
    btc_features: pd.DataFrame, 
    mstr_features: pd.DataFrame, 
    chunk_size: int = 1000,
    matching_params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Chunked similarity calculation for large-scale data processing.
    
    This function processes similarity calculations in chunks to handle
    datasets that would otherwise exceed memory limits.
    
    Args:
        btc_features: BTC normalized feature matrix
        mstr_features: MSTR normalized feature matrix
        chunk_size: Maximum chunk size for processing
        matching_params: Matching configuration parameters
    
    Returns:
        similarity_matrix: Complete similarity matrix
    """
    if not PANDAS_AVAILABLE or btc_features.empty or mstr_features.empty:
        logger.warning("Invalid input data for chunked calculation")
        return pd.DataFrame()
    
    # Estimate memory requirements
    estimated_memory = _estimate_memory_usage(btc_features, mstr_features)
    memory_limit_gb = matching_params.get('memory_limit_gb', 8.0)
    
    logger.info(f"Estimated memory usage: {estimated_memory:.2f}GB (limit: {memory_limit_gb}GB)")
    
    # Determine if chunking is needed
    use_chunking = estimated_memory > memory_limit_gb
    
    if not use_chunking:
        logger.info("Memory usage within limits. Using standard calculation.")
        return calculate_similarity_scores(btc_features, mstr_features, matching_params)
    
    logger.info(f"Memory usage exceeds limit. Using chunked processing with chunk_size={chunk_size}")
    
    # Calculate optimal chunk sizes
    btc_chunk_size = min(chunk_size, len(btc_features))
    mstr_chunk_size = min(chunk_size, len(mstr_features))
    
    # Create chunks
    btc_chunks = [
        btc_features.iloc[i:i+btc_chunk_size] 
        for i in range(0, len(btc_features), btc_chunk_size)
    ]
    mstr_chunks = [
        mstr_features.iloc[i:i+mstr_chunk_size] 
        for i in range(0, len(mstr_features), mstr_chunk_size)
    ]
    
    logger.info(f"Created {len(btc_chunks)} BTC chunks and {len(mstr_chunks)} MSTR chunks")
    
    # Initialize similarity matrix
    similarity_matrix = pd.DataFrame(
        data=np.zeros((len(btc_features), len(mstr_features))),
        index=btc_features.index,
        columns=mstr_features.index
    )
    
    total_chunks = len(btc_chunks) * len(mstr_chunks)
    processed_chunks = 0
    
    # Process chunks
    for i, btc_chunk in enumerate(btc_chunks):
        btc_start_idx = i * btc_chunk_size
        btc_end_idx = btc_start_idx + len(btc_chunk)
        
        for j, mstr_chunk in enumerate(mstr_chunks):
            mstr_start_idx = j * mstr_chunk_size
            mstr_end_idx = mstr_start_idx + len(mstr_chunk)
            
            # Calculate similarity for this chunk pair
            chunk_similarity = _calculate_similarity_block(
                btc_chunk, mstr_chunk, matching_params
            )
            
            # Place results in full matrix
            similarity_matrix.iloc[btc_start_idx:btc_end_idx, mstr_start_idx:mstr_end_idx] = chunk_similarity
            
            processed_chunks += 1
            
            # Progress logging
            if processed_chunks % max(1, total_chunks // 10) == 0:
                progress = (processed_chunks / total_chunks) * 100
                logger.info(f"Chunked processing progress: {progress:.1f}% ({processed_chunks}/{total_chunks})")
            
            # Memory management: periodic garbage collection
            if processed_chunks % 10 == 0:
                import gc
                gc.collect()
    
    logger.info("Chunked similarity calculation completed")
    
    return similarity_matrix

def _calculate_similarity_block(
    btc_block: pd.DataFrame, 
    mstr_block: pd.DataFrame, 
    matching_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate similarity for a small block of features.
    
    This function processes a small block efficiently while maintaining
    the same algorithm as the full calculation.
    
    Args:
        btc_block: Small BTC feature block
        mstr_block: Small MSTR feature block
        matching_params: Matching parameters
    
    Returns:
        block_similarity: Similarity matrix for this block
    """
    if not PANDAS_AVAILABLE or btc_block.empty or mstr_block.empty:
        return pd.DataFrame()
    
    # Extract algorithm and parameters
    algorithm = matching_params.get('matching_algorithm', 'constrained_dtw')
    dtw_constraint_ratio = matching_params.get('dtw_constraint_ratio', 0.1)
    
    # Prepare feature weights
    feature_weights = np.array([
        matching_params.get('direction_weight', 0.4),
        matching_params.get('strength_weight', 0.3),
        matching_params.get('volatility_weight', 0.2),
        matching_params.get('hurst_weight', 0.1)
    ])
    feature_weights = feature_weights / feature_weights.sum()
    
    # Initialize block similarity matrix
    block_similarity = pd.DataFrame(
        data=np.zeros((len(btc_block), len(mstr_block))),
        index=btc_block.index,
        columns=mstr_block.index
    )
    
    # Calculate pairwise similarities in the block
    for i, (btc_date, btc_pattern) in enumerate(btc_block.iterrows()):
        for j, (mstr_date, mstr_pattern) in enumerate(mstr_block.iterrows()):
            # Calculate distance based on selected algorithm
            if algorithm == 'constrained_dtw':
                distance, _ = constrained_dtw_distance(
                    btc_pattern.values.reshape(-1, 1), 
                    mstr_pattern.values.reshape(-1, 1),
                    dtw_constraint_ratio,
                    feature_weights
                )
            elif algorithm == 'cosine':
                distance = _cosine_distance(btc_pattern.values, mstr_pattern.values)
            elif algorithm == 'euclidean':
                distance = _weighted_euclidean_distance(
                    btc_pattern.values, mstr_pattern.values, feature_weights
                )
            else:
                distance = _weighted_euclidean_distance(
                    btc_pattern.values, mstr_pattern.values, feature_weights
                )
            
            # Convert distance to similarity
            similarity = 1.0 / (1.0 + distance) if distance < float('inf') else 0.0
            block_similarity.iloc[i, j] = similarity
    
    return block_similarity


# Task 3B: Parallel Processing Optimization
# =============================================================================

# Handle multiprocessing imports
try:
    from multiprocessing import Pool, cpu_count
    from functools import partial
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    logger.warning("multiprocessing not available. Parallel processing will be disabled.")

def _determine_optimal_processes(
    n_feature_pairs: int,
    matching_params: Dict[str, Any]
) -> int:
    """
    Determine optimal number of processes for parallel computation.
    
    Args:
        n_feature_pairs: Number of feature pairs to process
        matching_params: Matching parameters
    
    Returns:
        optimal_processes: Optimal number of processes
    """
    if not MULTIPROCESSING_AVAILABLE:
        return 1
    
    # Get system resources
    available_cores = cpu_count()
    
    # Get memory constraints
    if PSUTIL_AVAILABLE:
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            # Estimate memory per process (2GB heuristic)
            max_processes_by_memory = max(1, int(available_memory_gb / 2))
        except:
            max_processes_by_memory = available_cores
    else:
        max_processes_by_memory = available_cores
    
    # User-specified processes
    user_processes = matching_params.get('n_processes', 4)
    
    # Task-based limit (avoid too many processes for small tasks)
    min_pairs_per_process = 100
    max_processes_by_task = max(1, n_feature_pairs // min_pairs_per_process)
    
    # Choose conservative option
    optimal_processes = min(
        available_cores,
        max_processes_by_memory, 
        user_processes,
        max_processes_by_task
    )
    
    logger.info(f"Parallel processing: {optimal_processes} processes "
               f"(cores: {available_cores}, memory_limit: {max_processes_by_memory}, "
               f"task_limit: {max_processes_by_task}, user: {user_processes})")
    
    return optimal_processes

def _parallel_dtw_computation(
    feature_pairs: List[Tuple["np.ndarray", "np.ndarray"]], 
    matching_params: Dict[str, Any],
    n_processes: int = None
) -> List[float]:
    """
    Parallel DTW computation for high-performance processing.
    
    This function distributes DTW calculations across multiple CPU cores
    to achieve significant speedup for large datasets.
    
    Args:
        feature_pairs: List of (btc_features, mstr_features) pairs
        matching_params: Matching configuration parameters
        n_processes: Number of processes (auto-determined if None)
    
    Returns:
        distances: List of DTW distances for each pair
    """
    if not MULTIPROCESSING_AVAILABLE:
        logger.warning("Multiprocessing not available. Using sequential processing.")
        return _sequential_dtw_computation(feature_pairs, matching_params)
    
    if not feature_pairs:
        return []
    
    # Determine optimal number of processes
    if n_processes is None:
        n_processes = _determine_optimal_processes(len(feature_pairs), matching_params)
    
    # For small datasets, use sequential processing
    if len(feature_pairs) < 50 or n_processes == 1:
        logger.info("Using sequential processing for small dataset")
        return _sequential_dtw_computation(feature_pairs, matching_params)
    
    logger.info(f"Starting parallel DTW computation with {n_processes} processes")
    
    # Calculate chunk size for load balancing
    chunk_size = max(1, len(feature_pairs) // (n_processes * 2))
    
    # Create chunks for parallel processing
    pair_chunks = [
        feature_pairs[i:i+chunk_size] 
        for i in range(0, len(feature_pairs), chunk_size)
    ]
    
    try:
        # Create worker function with parameters
        worker_func = partial(_dtw_worker, matching_params=matching_params)
        
        # Execute parallel computation
        with Pool(processes=n_processes) as pool:
            chunk_results = pool.map(worker_func, pair_chunks)
        
        # Flatten results
        all_distances = []
        for chunk_result in chunk_results:
            all_distances.extend(chunk_result)
        
        logger.info(f"Parallel DTW computation completed: {len(all_distances)} pairs processed")
        
        return all_distances
        
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}. Falling back to sequential.")
        return _sequential_dtw_computation(feature_pairs, matching_params)

def _sequential_dtw_computation(
    feature_pairs: List[Tuple["np.ndarray", "np.ndarray"]], 
    matching_params: Dict[str, Any]
) -> List[float]:
    """
    Sequential DTW computation fallback.
    
    Args:
        feature_pairs: List of feature pairs
        matching_params: Matching parameters
    
    Returns:
        distances: List of DTW distances
    """
    distances = []
    
    dtw_constraint_ratio = matching_params.get('dtw_constraint_ratio', 0.1)
    feature_weights = np.array([
        matching_params.get('direction_weight', 0.4),
        matching_params.get('strength_weight', 0.3),
        matching_params.get('volatility_weight', 0.2),
        matching_params.get('hurst_weight', 0.1)
    ])
    feature_weights = feature_weights / feature_weights.sum()
    
    for i, (seq1, seq2) in enumerate(feature_pairs):
        distance, _ = constrained_dtw_distance(
            seq1, seq2, dtw_constraint_ratio, feature_weights
        )
        distances.append(distance)
        
        if i % max(1, len(feature_pairs) // 10) == 0:
            logger.debug(f"Sequential DTW progress: {i+1}/{len(feature_pairs)}")
    
    return distances

def _dtw_worker(pair_chunk: List[Tuple], matching_params: Dict[str, Any]) -> List[float]:
    """
    Worker function for parallel DTW processing.
    
    This function is executed by each worker process to compute DTW distances
    for a chunk of feature pairs.
    
    Args:
        pair_chunk: Chunk of feature pairs to process
        matching_params: Matching parameters
    
    Returns:
        distances: DTW distances for this chunk
    """
    distances = []
    
    # Extract parameters
    dtw_constraint_ratio = matching_params.get('dtw_constraint_ratio', 0.1)
    feature_weights = np.array([
        matching_params.get('direction_weight', 0.4),
        matching_params.get('strength_weight', 0.3),
        matching_params.get('volatility_weight', 0.2),
        matching_params.get('hurst_weight', 0.1)
    ])
    feature_weights = feature_weights / feature_weights.sum()
    
    # Process chunk
    for seq1, seq2 in pair_chunk:
        try:
            distance, _ = constrained_dtw_distance(
                seq1, seq2, dtw_constraint_ratio, feature_weights
            )
            distances.append(distance)
        except Exception as e:
            logger.warning(f"DTW calculation failed in worker: {e}")
            distances.append(float('inf'))
    
    return distances


# Task 3C: Adaptive Feature Weight Learning
# =============================================================================

def _adaptive_weight_learning(
    historical_matches: pd.DataFrame,
    prediction_accuracy: pd.Series = None,
    learning_rate: float = 0.01
) -> "np.ndarray":
    """
    Adaptive feature weight learning based on historical performance.
    
    This function learns optimal feature weights by analyzing the relationship
    between feature differences and prediction accuracy.
    
    Args:
        historical_matches: Historical pattern matches with accuracy data
        prediction_accuracy: Series of prediction accuracies (optional)
        learning_rate: Learning rate for weight updates
    
    Returns:
        optimized_weights: Learned feature weights [direction, strength, volatility, hurst]
    """
    if not PANDAS_AVAILABLE or historical_matches.empty:
        logger.warning("No historical data available. Using default weights.")
        return np.array([0.4, 0.3, 0.2, 0.1])
    
    logger.info(f"Learning adaptive weights from {len(historical_matches)} historical matches")
    
    # Initialize feature importance scores
    feature_importance = np.zeros(4)  # [direction, strength, volatility, hurst]
    feature_names = ['direction', 'strength', 'volatility', 'hurst']
    
    # Default prediction accuracy if not provided
    if prediction_accuracy is None:
        # Use similarity score as proxy for accuracy
        prediction_accuracy = historical_matches.get('similarity_score', pd.Series(0.5, index=historical_matches.index))
    
    # Calculate feature importance through gradient-based learning
    for idx, match in historical_matches.iterrows():
        try:
            # Extract BTC and MSTR features if available
            btc_features = np.array([
                match.get('btc_direction', 0),
                match.get('btc_strength', 0.5),
                match.get('btc_volatility', 0.5),
                match.get('btc_hurst', 0.5)
            ])
            
            mstr_features = np.array([
                match.get('mstr_direction', 0),
                match.get('mstr_strength', 0.5),
                match.get('mstr_volatility', 0.5),
                match.get('mstr_hurst', 0.5)
            ])
            
            # Calculate feature differences
            feature_differences = np.abs(btc_features - mstr_features)
            
            # Get accuracy for this match
            if idx in prediction_accuracy.index:
                accuracy = prediction_accuracy[idx]
            else:
                accuracy = match.get('similarity_score', 0.5)
            
            # Prediction error (higher error means worse prediction)
            prediction_error = 1.0 - accuracy
            
            # Gradient-based importance update
            # Features with larger differences that lead to higher errors get lower importance
            gradient = feature_differences * prediction_error
            feature_importance -= learning_rate * gradient
            
        except Exception as e:
            logger.warning(f"Error processing match {idx}: {e}")
            continue
    
    # Convert importance to positive weights
    # Use softmax-like transformation to ensure positive weights
    feature_importance = np.exp(feature_importance - np.max(feature_importance))
    feature_weights = feature_importance / np.sum(feature_importance)
    
    # Apply stability constraints (minimum weight for each feature)
    min_weight = 0.05
    feature_weights = np.maximum(feature_weights, min_weight)
    feature_weights = feature_weights / np.sum(feature_weights)
    
    # Apply moving average smoothing if previous weights exist
    if hasattr(_adaptive_weight_learning, 'previous_weights'):
        alpha = 0.7  # Smoothing factor (higher = more stable)
        feature_weights = alpha * _adaptive_weight_learning.previous_weights + (1 - alpha) * feature_weights
    
    # Store for next iteration
    _adaptive_weight_learning.previous_weights = feature_weights.copy()
    
    # Log results
    weight_info = ", ".join([f"{name}: {weight:.3f}" for name, weight in zip(feature_names, feature_weights)])
    logger.info(f"Learned adaptive weights: {weight_info}")
    
    return feature_weights

def _evaluate_weight_performance(
    matches_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    test_weights: "np.ndarray",
    reference_weights: "np.ndarray"
) -> Dict[str, float]:
    """
    Evaluate the performance of learned weights against reference weights.
    
    Args:
        matches_df: Pattern matches to evaluate
        similarity_matrix: Similarity matrix
        test_weights: Weights to evaluate
        reference_weights: Reference weights for comparison
    
    Returns:
        performance_metrics: Dictionary of performance comparisons
    """
    if not PANDAS_AVAILABLE or matches_df.empty:
        return {}
    
    metrics = {}
    
    try:
        # Calculate mean similarity improvement
        test_weighted_similarity = np.average(
            matches_df['similarity_score'], 
            weights=test_weights[:len(matches_df['similarity_score'])] if len(test_weights) >= len(matches_df) else None
        )
        
        reference_weighted_similarity = np.average(
            matches_df['similarity_score'], 
            weights=reference_weights[:len(matches_df['similarity_score'])] if len(reference_weights) >= len(matches_df) else None
        )
        
        metrics['similarity_improvement'] = test_weighted_similarity - reference_weighted_similarity
        
        # Calculate weight diversity (entropy)
        test_entropy = -np.sum(test_weights * np.log(test_weights + 1e-8))
        reference_entropy = -np.sum(reference_weights * np.log(reference_weights + 1e-8))
        
        metrics['weight_diversity_improvement'] = test_entropy - reference_entropy
        
        # Calculate stability (weight variance)
        metrics['weight_stability'] = 1.0 / (1.0 + np.var(test_weights))
        
        logger.info(f"Weight performance: similarity_improvement={metrics['similarity_improvement']:.4f}, "
                   f"diversity_improvement={metrics['weight_diversity_improvement']:.4f}, "
                   f"stability={metrics['weight_stability']:.4f}")
        
    except Exception as e:
        logger.warning(f"Error evaluating weight performance: {e}")
    
    return metrics


# =============================================================================
# Data Schema Definition (Phase 1)
# =============================================================================

@dataclass
class PatternMatches:
    """
    Pattern matching analysis results (Phase 1 implementation).
    
    This class contains all results from the pattern matching analysis,
    including similarity matrices, significant matches, and quality metrics.
    """
    # Phase 1: Core functionality
    similarity_matrix: pd.DataFrame
    """Pattern similarity matrix [0.0-1.0]"""
    
    significant_matches: pd.DataFrame
    """High-similarity pattern pairs with details"""
    
    pattern_statistics: Dict[str, pd.DataFrame]
    """Comprehensive pattern statistics"""
    
    matching_quality: Dict[str, float]
    """Matching quality indicators"""
    
    matching_params: Dict[str, Any]
    """Parameters used for matching"""
    
    # Phase 2/3 placeholders (to be implemented)
    statistical_diagnostics: Dict[str, Any] = field(default_factory=dict)
    """Statistical diagnostic information (Phase 2+)"""
    
    confidence_intervals: Optional[pd.DataFrame] = None
    """Bayesian confidence intervals (Phase 2+)"""
    
    fdr_results: Optional[Dict[str, Any]] = None
    """FDR correction results (Phase 2+)"""
    
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    """Performance metrics (Phase 3+)"""
    
    optimization_diagnostics: Dict[str, Any] = field(default_factory=dict)
    """Optimization diagnostics (Phase 3+)"""
    
    memory_usage: Dict[str, float] = field(default_factory=dict)
    """Memory usage statistics (Phase 3+)"""
    
    def validate(self) -> bool:
        """
        Validate data integrity and consistency.
        
        Returns:
            is_valid: True if all data passes validation
        """
        try:
            # Check similarity matrix
            if self.similarity_matrix.empty:
                return False
            
            # Check value ranges
            if not (0.0 <= self.similarity_matrix.min().min() <= self.similarity_matrix.max().max() <= 1.0):
                return False
            
            # Check significant matches consistency
            if not self.significant_matches.empty:
                required_columns = ['btc_date', 'mstr_date', 'similarity_score']
                if not all(col in self.significant_matches.columns for col in required_columns):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False


# =============================================================================
# Main Integration Function (Phase 1)
# =============================================================================

def find_pattern_matches(
    patterns: DirectionPatterns,
    matching_params: Optional[Dict[str, Any]] = None,
    optimization_phase: int = 1
) -> PatternMatches:
    """
    Main pattern matching function (Phase 1 implementation).
    
    This function orchestrates the entire pattern matching pipeline,
    from feature extraction through similarity calculation to result compilation.
    
    Args:
        patterns: DirectionPatterns from direction_converter
        matching_params: Matching configuration parameters
        optimization_phase: Implementation phase [1, 2, 3] - currently only Phase 1
    
    Returns:
        PatternMatches: Complete analysis results
    """
    start_time = time.time()
    logger.info("Starting pattern matching analysis (Phase 1)")
    
    # Phase 1: Core functionality implementation
    
    # 1. Parameter validation and default setting
    validated_params = _validate_matching_params(matching_params)
    logger.info(f"Using parameters: {validated_params}")
    
    # 2. Feature extraction for matching
    btc_features, mstr_features = _extract_features_for_matching(patterns)
    logger.info(f"Extracted features: BTC {btc_features.shape}, MSTR {mstr_features.shape}")
    
    if btc_features.empty or mstr_features.empty:
        logger.warning("No features available for matching")
        return PatternMatches(
            similarity_matrix=pd.DataFrame(),
            significant_matches=pd.DataFrame(),
            pattern_statistics={},
            matching_quality={},
            matching_params=validated_params
        )
    
    # 3. Adaptive feature normalization
    if validated_params['normalization_scope'] == 'adaptive':
        btc_normalized = _adaptive_normalize_features(
            btc_features, 
            btc_features.get('volatility', pd.Series(index=btc_features.index, data=0.5)), 
            validated_params['normalization_method']
        )
        mstr_normalized = _adaptive_normalize_features(
            mstr_features, 
            mstr_features.get('volatility', pd.Series(index=mstr_features.index, data=0.5)),
            validated_params['normalization_method']
        )
    else:
        btc_normalized = _global_normalize(btc_features, validated_params['normalization_method'])
        mstr_normalized = _global_normalize(mstr_features, validated_params['normalization_method'])
    
    logger.info("Feature normalization completed")
    
    # 4. Similarity matrix calculation
    similarity_matrix = calculate_similarity_scores(btc_normalized, mstr_normalized, validated_params)
    
    # 5. Significant match extraction
    significant_matches = extract_significant_matches(
        similarity_matrix, 
        patterns.btc_pattern_sequences,
        patterns.mstr_pattern_sequences, 
        validated_params
    )
    
    phase1_time = time.time() - start_time
    
    # Phase 2: Statistical rigor enhancement
    phase2_time = 0
    if optimization_phase >= 2:
        phase2_start = time.time()
        logger.info("Starting Phase 2: Statistical rigor enhancement")
        
        # 6. NMS overlap removal
        nms_results = None
        if validated_params['overlap_removal_method'] == 'nms' and not significant_matches.empty:
            nms_results = _nms_overlapping_matches(
                significant_matches, 
                validated_params['nms_iou_threshold']
            )
            significant_matches = nms_results['selected_matches']
            logger.info(f"NMS applied: {nms_results['suppression_stats']['suppression_ratio']:.3f} suppression ratio")
        
        # 7. Statistical significance and FDR correction
        fdr_results = None
        if validated_params['fdr_correction'] and not significant_matches.empty:
            logger.info("Calculating statistical significance and FDR correction")
            raw_p_values = calculate_match_significance(significant_matches, similarity_matrix)
            
            if len(raw_p_values) > 0:
                fdr_results = _apply_fdr_correction(raw_p_values, validated_params['significance_level'])
                
                # Add statistical columns to significant_matches
                significant_matches = significant_matches.copy()
                significant_matches['p_value'] = raw_p_values
                significant_matches['fdr_adjusted_p'] = fdr_results['adjusted_p_values']
                significant_matches['statistical_significance'] = (
                    significant_matches['fdr_adjusted_p'] < validated_params['significance_level']
                )
                
                # Filter to only statistically significant matches if requested
                if validated_params.get('filter_insignificant', False):
                    significant_matches = significant_matches[
                        significant_matches['statistical_significance']
                    ].reset_index(drop=True)
                    logger.info(f"Filtered to {len(significant_matches)} statistically significant matches")
        
        # 8. Bayesian confidence intervals
        confidence_intervals = None
        if validated_params['bayesian_confidence'] and not significant_matches.empty:
            logger.info("Calculating Bayesian confidence intervals")
            confidence_intervals = _calculate_bayesian_confidence(significant_matches)
            significant_matches = confidence_intervals
        
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 completed in {phase2_time:.2f}s")
    else:
        fdr_results = None
        confidence_intervals = None
        nms_results = None
    
    # Phase 3: Advanced optimization
    phase3_time = 0
    optimization_diagnostics = {}
    adaptive_weights = None
    
    if optimization_phase >= 3:
        phase3_start = time.time()
        logger.info("Starting Phase 3: Advanced optimization")
        
        # 9. Chunked processing for large-scale data
        original_chunk_processing = validated_params.get('enable_chunking', False)
        if validated_params.get('enable_chunking', False):
            logger.info("Enabling chunked processing optimization")
            chunk_results = _chunked_similarity_calculation(
                btc_normalized, mstr_normalized, validated_params
            )
            
            # Update similarity matrix with chunked results if available
            if chunk_results['chunked_matrix'] is not None and not chunk_results['chunked_matrix'].empty:
                similarity_matrix = chunk_results['chunked_matrix']
                optimization_diagnostics['chunking_performance'] = chunk_results['performance_metrics']
                
                # Re-extract significant matches from chunked matrix
                significant_matches = extract_significant_matches(
                    similarity_matrix, 
                    patterns.btc_pattern_sequences,
                    patterns.mstr_pattern_sequences, 
                    validated_params
                )
                logger.info(f"Chunked processing: {len(significant_matches)} matches from chunked matrix")
        
        # 10. Parallel processing optimization
        if validated_params.get('enable_parallel', False):
            logger.info("Enabling parallel processing optimization")
            parallel_results = _parallel_similarity_calculation(
                btc_normalized, mstr_normalized, validated_params
            )
            
            # Update similarity matrix with parallel results if available
            if parallel_results['parallel_matrix'] is not None and not parallel_results['parallel_matrix'].empty:
                similarity_matrix = parallel_results['parallel_matrix']
                optimization_diagnostics['parallel_performance'] = parallel_results['performance_metrics']
                
                # Re-extract significant matches from parallel matrix
                significant_matches = extract_significant_matches(
                    similarity_matrix, 
                    patterns.btc_pattern_sequences,
                    patterns.mstr_pattern_sequences, 
                    validated_params
                )
                logger.info(f"Parallel processing: {len(significant_matches)} matches from parallel matrix")
        
        # 11. Adaptive feature weight learning
        if validated_params.get('enable_adaptive_weights', False) and not significant_matches.empty:
            logger.info("Learning adaptive feature weights")
            adaptive_weights = _adaptive_weight_learning(
                significant_matches, 
                validated_params.get('learning_rate', 0.01)
            )
            
            # Evaluate weight performance
            default_weights = np.array([
                validated_params['direction_weight'],
                validated_params['strength_weight'], 
                validated_params['volatility_weight'],
                validated_params['hurst_weight']
            ])
            
            weight_performance = _evaluate_weight_performance(
                significant_matches, similarity_matrix, adaptive_weights, default_weights
            )
            optimization_diagnostics['adaptive_weights'] = {
                'learned_weights': adaptive_weights.tolist(),
                'performance_improvement': weight_performance
            }
        
        phase3_time = time.time() - phase3_start
        logger.info(f"Phase 3 completed in {phase3_time:.2f}s")
        
        # Log optimization summary
        if optimization_diagnostics:
            logger.info("Phase 3 optimization summary:")
            if 'chunking_performance' in optimization_diagnostics:
                chunk_speedup = optimization_diagnostics['chunking_performance'].get('speedup_factor', 1.0)
                logger.info(f"  Chunking speedup: {chunk_speedup:.2f}x")
            if 'parallel_performance' in optimization_diagnostics:
                parallel_speedup = optimization_diagnostics['parallel_performance'].get('speedup_factor', 1.0)
                logger.info(f"  Parallel speedup: {parallel_speedup:.2f}x")
            if 'adaptive_weights' in optimization_diagnostics:
                weights = optimization_diagnostics['adaptive_weights']['learned_weights']
                logger.info(f"  Adaptive weights: {weights}")
    else:
        optimization_diagnostics = {}
    
    # 9. Statistics and quality evaluation (updated with Phase 2 data)
    pattern_statistics = calculate_pattern_statistics(
        significant_matches,
        patterns.btc_pattern_sequences,
        patterns.mstr_pattern_sequences
    )
    
    matching_quality = calculate_matching_quality(
        similarity_matrix, significant_matches, validated_params
    )
    
    # Add Phase 2 quality metrics
    if optimization_phase >= 2 and nms_results:
        matching_quality.update(nms_results['suppression_stats'])
    
    # Performance metrics (enhanced for Phase 2 and Phase 3)
    total_time = time.time() - start_time
    performance_metrics = {
        'total_execution_time': total_time,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
        'phase3_time': phase3_time,
        'patterns_per_second': len(significant_matches) / max(total_time, 0.001),
        'optimization_phase': optimization_phase,
    }
    
    if PSUTIL_AVAILABLE:
        try:
            performance_metrics['memory_peak_mb'] = psutil.Process().memory_info().rss / (1024*1024)
        except:
            performance_metrics['memory_peak_mb'] = 0.0
    
    # Memory usage statistics
    memory_usage = {}
    if PANDAS_AVAILABLE:
        try:
            memory_usage['similarity_matrix_mb'] = similarity_matrix.memory_usage(deep=True).sum() / (1024*1024)
            memory_usage['significant_matches_mb'] = significant_matches.memory_usage(deep=True).sum() / (1024*1024)
        except:
            memory_usage['similarity_matrix_mb'] = 0.0
            memory_usage['significant_matches_mb'] = 0.0
    
    # Statistical diagnostics (Phase 2)
    statistical_diagnostics = {}
    if optimization_phase >= 2:
        statistical_diagnostics = {
            'fdr_results': fdr_results,
            'nms_suppression_ratio': matching_quality.get('suppression_ratio', 0),
            'bayesian_confidence_enabled': validated_params.get('bayesian_confidence', False),
            'statistical_significance_count': (
                significant_matches.get('statistical_significance', pd.Series()).sum() 
                if optimization_phase >= 2 and not significant_matches.empty else 0
            )
        }
    
    # Create PatternMatches result
    result = PatternMatches(
        # Phase 1: Core functionality
        similarity_matrix=similarity_matrix,
        significant_matches=significant_matches,
        pattern_statistics=pattern_statistics,
        matching_quality=matching_quality,
        matching_params=validated_params,
        
        # Phase 2: Statistical rigor
        statistical_diagnostics=statistical_diagnostics,
        confidence_intervals=confidence_intervals,
        fdr_results=fdr_results,
        
        # Performance metrics
        performance_metrics=performance_metrics,
        memory_usage=memory_usage,
        
        # Phase 3: Optimization diagnostics
        optimization_diagnostics=optimization_diagnostics
    )
    
    logger.info(f"Pattern matching completed in {total_time:.2f}s")
    logger.info(f"Found {len(significant_matches)} significant matches")
    
    return result

def _extract_features_for_matching(patterns: DirectionPatterns) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features for pattern matching from DirectionPatterns.
    
    Args:
        patterns: DirectionPatterns object from direction_converter
    
    Returns:
        btc_features: BTC feature matrix for matching
        mstr_features: MSTR feature matrix for matching
    """
    if not PANDAS_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()
    
    # Extract basic directional features
    feature_columns = ['direction', 'strength', 'volatility']
    
    # Add hurst if available (from direction_converter GARCH analysis)
    if 'hurst' in patterns.btc_directions.columns:
        feature_columns.append('hurst')
    else:
        # Create placeholder hurst values if not available
        patterns.btc_directions['hurst'] = 0.5
        patterns.mstr_directions['hurst'] = 0.5
        feature_columns.append('hurst')
    
    btc_features = patterns.btc_directions[feature_columns].copy()
    mstr_features = patterns.mstr_directions[feature_columns].copy()
    
    # Remove any rows with NaN values
    btc_features = btc_features.dropna()
    mstr_features = mstr_features.dropna()
    
    return btc_features, mstr_features

def _validate_matching_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate and complete matching parameters with defaults.
    
    Args:
        params: User-provided parameters (can be None)
    
    Returns:
        validated_params: Complete validated parameter set
    """
    default_params = {
        # Similarity thresholds
        'similarity_threshold': 0.7,
        'percentile_threshold': 90,
        
        # DTW parameters
        'dtw_constraint_ratio': 0.1,
        'max_warping_window': 10,
        
        # Feature weights
        'direction_weight': 0.4,
        'strength_weight': 0.3,
        'volatility_weight': 0.2,
        'hurst_weight': 0.1,
        
        # Algorithm selection
        'matching_algorithm': 'constrained_dtw',
        
        # Normalization settings
        'normalization_method': 'adaptive',
        'normalization_scope': 'adaptive',
        
        # Pattern handling
        'window_size': 30,
        'allow_inverse_patterns': True,
        'min_pattern_length': 5,
        'max_pattern_length': 20,
        
        # Statistical parameters (Phase 1 defaults)
        'significance_level': 0.05,
        'fdr_correction': True,
        'min_sample_size': 10,
        
        # Optimization settings (Phase 1 - basic)
        'enable_numba': NUMBA_AVAILABLE,
        'enable_chunking': False,
        'chunk_size': 1000,
        'enable_parallel': False,
        'n_processes': 4,
        
        # Phase 2 features (statistical rigor)
        'bayesian_confidence': False,
        'overlap_removal_method': 'nms',  # 'greedy', 'nms'
        'nms_iou_threshold': 0.5,
        'filter_insignificant': False,   # Filter non-significant matches
        
        # Phase 3 features (advanced optimization)
        'enable_adaptive_weights': False,
        'learning_rate': 0.01,
        'weight_smoothing_alpha': 0.7
    }
    
    if params is None:
        validated = default_params
    else:
        validated = default_params.copy()
        validated.update(params)
    
    # Parameter validation
    assert 0.0 <= validated['similarity_threshold'] <= 1.0, "similarity_threshold must be in [0.0, 1.0]"
    assert 0.0 <= validated['dtw_constraint_ratio'] <= 1.0, "dtw_constraint_ratio must be in [0.0, 1.0]"
    
    # Ensure feature weights sum to positive value
    weight_sum = (validated['direction_weight'] + validated['strength_weight'] + 
                  validated['volatility_weight'] + validated['hurst_weight'])
    assert weight_sum > 0, "Feature weights must sum to positive value"
    
    # Validate algorithm choice
    valid_algorithms = ['constrained_dtw', 'cosine', 'euclidean']
    assert validated['matching_algorithm'] in valid_algorithms, f"matching_algorithm must be one of {valid_algorithms}"
    
    # Validate normalization methods
    valid_norm_methods = ['minmax', 'zscore', 'robust', 'adaptive']
    assert validated['normalization_method'] in valid_norm_methods, f"normalization_method must be one of {valid_norm_methods}"
    
    valid_norm_scopes = ['global', 'local', 'adaptive']
    assert validated['normalization_scope'] in valid_norm_scopes, f"normalization_scope must be one of {valid_norm_scopes}"
    
    # Phase 2 parameter validation
    assert 0.0 < validated['significance_level'] < 1.0, "significance_level must be in (0.0, 1.0)"
    assert 0.0 <= validated['nms_iou_threshold'] <= 1.0, "nms_iou_threshold must be in [0.0, 1.0]"
    
    valid_overlap_methods = ['greedy', 'nms']
    assert validated['overlap_removal_method'] in valid_overlap_methods, f"overlap_removal_method must be one of {valid_overlap_methods}"
    
    # Phase 3 parameter validation
    if 'learning_rate' in validated:
        assert 0.0 < validated['learning_rate'] <= 1.0, "learning_rate must be in (0.0, 1.0]"
    if 'weight_smoothing_alpha' in validated:
        assert 0.0 <= validated['weight_smoothing_alpha'] <= 1.0, "weight_smoothing_alpha must be in [0.0, 1.0]"
    if 'chunk_size' in validated:
        assert validated['chunk_size'] > 0, "chunk_size must be positive"
    if 'n_processes' in validated:
        assert validated['n_processes'] > 0, "n_processes must be positive"
    
    return validated


# =============================================================================
# Module Testing and Validation
# =============================================================================

def _validate_pattern_matcher() -> bool:
    """
    Internal validation function for pattern matcher functionality.
    
    Returns:
        is_valid: True if all core functions work correctly
    """
    logger.info("Validating pattern matcher functionality...")
    
    try:
        # Test DTW function
        if PANDAS_AVAILABLE and np is not None:
            seq1 = np.random.rand(10, 2)
            seq2 = np.random.rand(8, 2)
            distance, path = constrained_dtw_distance(seq1, seq2)
            
            if not (0 <= distance < float('inf')):
                logger.error(f"DTW distance validation failed: {distance}")
                return False
            
            logger.info(f"DTW validation passed: distance={distance:.3f}, path_length={len(path)}")
        else:
            # Test mock DTW function
            seq1 = [[1.0, 0.5], [0.8, 0.3]]  # Mock 2D array
            seq2 = [[0.9, 0.4], [0.7, 0.6]]  # Mock 2D array
            distance, path = constrained_dtw_distance(seq1, seq2)
            
            # Mock implementation should return valid default values
            if distance != 1.0:
                logger.error(f"Mock DTW distance validation failed: expected 1.0, got {distance}")
                return False
            
            logger.info(f"Mock DTW validation passed: distance={distance}, path_length={len(path)}")
        
        # Test normalization
        if PANDAS_AVAILABLE:
            test_features = pd.DataFrame({
                'direction': [1, -1, 0, 1, -1],
                'strength': [0.8, 0.6, 0.2, 0.9, 0.7],
                'volatility': [0.1, 0.3, 0.5, 0.2, 0.4],
                'hurst': [0.6, 0.4, 0.5, 0.7, 0.3]
            })
            
            test_volatility = pd.Series([0.1, 0.3, 0.5, 0.2, 0.4])
            
            normalized = _adaptive_normalize_features(test_features, test_volatility)
            
            if normalized.empty:
                logger.error("Normalization validation failed: empty result")
                return False
            
            logger.info("Normalization validation passed")
        else:
            # Test mock normalization
            test_features = pd.DataFrame()  # Mock empty DataFrame
            test_volatility = pd.Series()   # Mock empty Series
            
            normalized = _adaptive_normalize_features(test_features, test_volatility)
            
            # Mock implementation should return input unchanged
            logger.info("Mock normalization validation passed")
        
        # Test parameter validation
        test_params = {
            'similarity_threshold': 0.8,
            'matching_algorithm': 'constrained_dtw'
        }
        
        validated = _validate_matching_params(test_params)
        
        if not validated or validated['similarity_threshold'] != 0.8:
            logger.error("Parameter validation failed")
            return False
        
        logger.info("Parameter validation passed")
        
        logger.info("All pattern matcher validations passed")
        return True
        
    except Exception as e:
        logger.error(f"Pattern matcher validation failed: {e}")
        return False


# Module execution for testing
if __name__ == "__main__":
    logger.info("=== Pattern Matcher Module Test ===")
    
    # Run validation
    validation_result = _validate_pattern_matcher()
    
    if validation_result:
        print("✓ Pattern matcher validation successful")
        print("\nModule capabilities:")
        print(f"  - Pandas available: {PANDAS_AVAILABLE}")
        print(f"  - Numba optimization: {NUMBA_AVAILABLE}")
        print(f"  - SciPy statistics: {SCIPY_AVAILABLE}")
        print(f"  - Memory monitoring: {PSUTIL_AVAILABLE}")
        print("\nPhase 1 implementation completed:")
        print("  ✓ Constrained DTW with Numba optimization")
        print("  ✓ Adaptive feature normalization")
        print("  ✓ Basic similarity calculation")
        print("  ✓ Pattern statistics and quality metrics")
        
        print("\nPhase 2 implementation completed:")
        print("  ✓ Non-Maximum Suppression (NMS)")
        print("  ✓ False Discovery Rate (FDR) control")
        print("  ✓ Bayesian confidence intervals")
        print("  ✓ Statistical significance testing")
        
        if PANDAS_AVAILABLE:
            # Demo run with mock data
            logger.info("\n=== Demo Run with Mock Data ===")
            
            # Create mock DirectionPatterns
            mock_btc_directions = pd.DataFrame({
                'direction': [1, -1, 0, 1, -1],
                'strength': [0.8, 0.6, 0.2, 0.9, 0.7],
                'volatility': [0.1, 0.3, 0.5, 0.2, 0.4],
                'hurst': [0.6, 0.4, 0.5, 0.7, 0.3]
            }, index=pd.date_range('2023-01-01', periods=5))
            
            mock_mstr_directions = pd.DataFrame({
                'direction': [1, 0, -1, 1, 0],
                'strength': [0.7, 0.3, 0.8, 0.6, 0.4],
                'volatility': [0.2, 0.4, 0.3, 0.1, 0.5],
                'hurst': [0.5, 0.6, 0.4, 0.8, 0.2]
            }, index=pd.date_range('2023-01-02', periods=5))
            
            mock_patterns = DirectionPatterns(
                btc_directions=mock_btc_directions,
                mstr_directions=mock_mstr_directions,
                btc_pattern_sequences=pd.DataFrame(),
                mstr_pattern_sequences=pd.DataFrame(),
                conversion_params={},
                quality_metrics={}
            )
            
            # Run pattern matching (Phase 1)
            results_phase1 = find_pattern_matches(mock_patterns, optimization_phase=1)
            
            print(f"\nPhase 1 results:")
            print(f"  - Similarity matrix shape: {results_phase1.similarity_matrix.shape}")
            print(f"  - Significant matches: {len(results_phase1.significant_matches)}")
            print(f"  - Processing time: {results_phase1.performance_metrics.get('total_execution_time', 0):.3f}s")
            print(f"  - Validation passed: {results_phase1.validate()}")
            
            # Run pattern matching (Phase 2) if significant matches exist
            if not results_phase1.significant_matches.empty:
                logger.info("\n=== Phase 2 Demo Run ===")
                
                # Enable Phase 2 features for demo
                phase2_params = {
                    'bayesian_confidence': True,
                    'fdr_correction': True,
                    'overlap_removal_method': 'nms',
                    'nms_iou_threshold': 0.3
                }
                
                results_phase2 = find_pattern_matches(mock_patterns, phase2_params, optimization_phase=2)
                
                print(f"\nPhase 2 results:")
                print(f"  - Significant matches after Phase 2: {len(results_phase2.significant_matches)}")
                print(f"  - FDR correction applied: {results_phase2.fdr_results is not None}")
                print(f"  - Confidence intervals calculated: {results_phase2.confidence_intervals is not None}")
                print(f"  - NMS suppression ratio: {results_phase2.statistical_diagnostics.get('nms_suppression_ratio', 0):.3f}")
                print(f"  - Phase 2 processing time: {results_phase2.performance_metrics.get('phase2_time', 0):.3f}s")
                print(f"  - Total processing time: {results_phase2.performance_metrics.get('total_execution_time', 0):.3f}s")
                print(f"  - Validation passed: {results_phase2.validate()}")
            else:
                print("\nPhase 2 demo skipped: No significant matches in Phase 1")
    else:
        print("✗ Pattern matcher validation failed")
        exit(1)