"""
analysis/pattern_analysis/direction_converter.py

Direction converter module for MSTR prediction system.

This module transforms price time series data into multi-dimensional feature vectors 
that capture dynamic market states. It uses GARCH model volatility prediction as the 
core component, combined with trend momentum, reaction speed, and persistence to 
quantify the market from multiple angles.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass
import logging

# Handle imports with fallbacks for testing
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Mock classes for basic testing
    class MockPandas:
        class DataFrame:
            def __init__(self):
                self.empty = True
            def __len__(self):
                return 0
        class Series:
            def __init__(self):
                pass
        class Timestamp:
            def __init__(self, dt):
                self.dt = dt
            @staticmethod
            def now():
                return MockPandas.Timestamp(datetime.now())
    pd = MockPandas()
    np = None

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch package not available. GARCH functionality will be disabled.")

try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False
    warnings.warn("scipy/sklearn not available. Some statistical functions will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from data.preprocessor import ProcessedDataContainer
except ImportError:
    logger.warning("ProcessedDataContainer not available. Using mock implementation.")
    # Mock implementation for testing
    @dataclass
    class ProcessedDataContainer:
        btc_processed: pd.DataFrame
        mstr_processed: pd.DataFrame
        gold_processed: pd.DataFrame
        common_date_range: Tuple[pd.Timestamp, pd.Timestamp]
        data_quality_report: Dict[str, Any]
        processing_metadata: Dict[str, Any]


# ===========================================
# CUSTOM EXCEPTIONS
# ===========================================

class DirectionConverterError(Exception):
    """Direction conversion error"""
    pass

class GARCHConvergenceError(Exception):
    """GARCH convergence error"""
    pass

class DataInsufficientError(Exception):
    """Insufficient data error"""
    pass


# ===========================================
# DATA CLASSES
# ===========================================

@dataclass
class DirectionPatterns:
    """Direction pattern analysis result"""
    
    btc_directions: pd.DataFrame
    """
    BTC direction data
    
    Required column structure:
    - index: pd.DatetimeIndex (daily)
    - columns: ['direction', 'strength', 'volatility', 'hurst', 'trend_duration']
    
    Data types:
    - direction: int {-1: down, 0: sideways, 1: up}
    - strength: float [0.0-1.0] (direction strength)
    - volatility: float [0.0-inf] (daily volatility)
    - hurst: float [0.0-1.0] (Hurst exponent)
    - trend_duration: int (consecutive days in same direction)
    
    Constraints:
    - direction in {-1, 0, 1}
    - strength, hurst in [0.0-1.0] range
    - trend_duration >= 1
    - no missing values
    """
    
    mstr_directions: pd.DataFrame
    """MSTR direction data (same column structure as btc_directions)"""
    
    btc_pattern_sequences: pd.DataFrame
    """
    BTC higher-order patterns (consecutive direction combinations)
    
    Required column structure:
    - index: pd.DatetimeIndex (pattern end date)
    - columns: ['pattern_length', 'pattern_code', 'pattern_strength', 'start_date']
    
    Data types:
    - pattern_length: int (pattern days: 3-10)
    - pattern_code: str (e.g., "110", "-1-10", "001")
    - pattern_strength: float [0.0-1.0] (pattern clarity)
    - start_date: pd.Timestamp (pattern start date)
    
    Constraints:
    - pattern_length in [3, 10] range
    - pattern_code is concatenated directions (-1,0,1)
    - start_date <= index (end date)
    - no overlapping patterns
    """
    
    mstr_pattern_sequences: pd.DataFrame
    """MSTR higher-order patterns (same column structure as btc_pattern_sequences)"""
    
    conversion_params: Dict[str, Any]
    """Conversion parameters used (for reproducibility)"""
    
    quality_metrics: Dict[str, float]
    """
    Conversion quality indicators
    {
        'pattern_coverage': float [0.0-1.0],      # Pattern detection coverage
        'avg_pattern_strength': float [0.0-1.0],  # Average pattern strength
        'data_completeness': float [0.0-1.0],     # Data completeness
        'direction_consistency': float [0.0-1.0], # Direction consistency
        'volatility_adaptation': float [0.0-1.0], # Volatility adaptation
    }
    """
    
    def validate(self) -> bool:
        """Data integrity validation"""
        checks = [
            self._validate_directions(self.btc_directions, 'BTC'),
            self._validate_directions(self.mstr_directions, 'MSTR'),
            self._validate_patterns(self.btc_pattern_sequences, 'BTC'),
            self._validate_patterns(self.mstr_pattern_sequences, 'MSTR'),
            self._validate_quality_metrics()
        ]
        return all(checks)
    
    def _validate_directions(self, df: pd.DataFrame, asset_name: str) -> bool:
        """Validate individual direction DataFrame"""
        try:
            required_columns = ['direction', 'strength', 'volatility', 'hurst', 'trend_duration']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"{asset_name} directions missing required columns")
                return False
            
            # Check direction values
            if not df['direction'].isin([-1, 0, 1]).all():
                logger.error(f"{asset_name} directions contain invalid values")
                return False
            
            # Check strength and hurst ranges
            if not ((df['strength'] >= 0.0) & (df['strength'] <= 1.0)).all():
                logger.error(f"{asset_name} strength values out of range [0.0-1.0]")
                return False
            
            if not ((df['hurst'] >= 0.0) & (df['hurst'] <= 1.0)).all():
                logger.error(f"{asset_name} hurst values out of range [0.0-1.0]")
                return False
            
            # Check for missing values
            if df.isnull().any().any():
                logger.error(f"{asset_name} directions contain missing values")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Direction validation failed for {asset_name}: {e}")
            return False
    
    def _validate_patterns(self, df: pd.DataFrame, asset_name: str) -> bool:
        """Validate pattern sequences DataFrame"""
        try:
            required_columns = ['pattern_length', 'pattern_code', 'pattern_strength', 'start_date']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"{asset_name} patterns missing required columns")
                return False
            
            # Check pattern length range
            if not ((df['pattern_length'] >= 3) & (df['pattern_length'] <= 10)).all():
                logger.error(f"{asset_name} pattern lengths out of range [3-10]")
                return False
            
            # Check pattern strength range
            if not ((df['pattern_strength'] >= 0.0) & (df['pattern_strength'] <= 1.0)).all():
                logger.error(f"{asset_name} pattern strength out of range [0.0-1.0]")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Pattern validation failed for {asset_name}: {e}")
            return False
    
    def _validate_quality_metrics(self) -> bool:
        """Validate quality metrics"""
        try:
            required_metrics = [
                'pattern_coverage', 'avg_pattern_strength', 'data_completeness',
                'direction_consistency', 'volatility_adaptation'
            ]
            
            for metric in required_metrics:
                if metric not in self.quality_metrics:
                    logger.error(f"Missing quality metric: {metric}")
                    return False
                
                value = self.quality_metrics[metric]
                if not (0.0 <= value <= 1.0):
                    logger.error(f"Quality metric {metric} out of range [0.0-1.0]: {value}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Quality metrics validation failed: {e}")
            return False


# ===========================================
# TASK 1: TREND INDICATORS CALCULATION
# ===========================================

def calculate_ema_trend(prices: pd.Series, short_window: int, long_window: int) -> Dict[str, pd.Series]:
    """
    Calculate EMA-based fast reaction trend.
    
    Args:
        prices: Price series
        short_window: Short-term EMA window
        long_window: Long-term EMA window
        
    Returns:
        Dictionary containing EMA data and trend indicators
        {
            'ema_short': Short-term EMA series,
            'ema_long': Long-term EMA series, 
            'trend_direction': Trend direction {-1, 0, 1},
            'trend_strength': Trend strength [0.0-inf] (to be normalized later)
        }
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        if len(prices) < max(short_window, long_window):
            raise DataInsufficientError(f"Insufficient data: {len(prices)} < {max(short_window, long_window)}")
        
        # Calculate EMAs
        ema_short = prices.ewm(span=short_window, adjust=False).mean()
        ema_long = prices.ewm(span=long_window, adjust=False).mean()
        
        # Calculate trend direction
        ema_diff = ema_short - ema_long
        trend_direction = np.sign(ema_diff).fillna(0).astype(int)

        # Calculate trend strength (relative to long-term EMA)
        trend_strength = np.abs(ema_diff) / ema_long
        
        # Handle potential division by zero
        trend_strength = trend_strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return {
            'ema_short': ema_short,
            'ema_long': ema_long,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }
        
    except Exception as e:
        logger.error(f"EMA trend calculation failed: {e}")
        raise DirectionConverterError(f"EMA calculation failed: {e}")


def calculate_sma_trend(prices: pd.Series, short_window: int, long_window: int) -> Dict[str, pd.Series]:
    """
    Calculate SMA-based stable trend.
    
    Args:
        prices: Price series
        short_window: Short-term SMA window
        long_window: Long-term SMA window
        
    Returns:
        Dictionary containing SMA data and trend indicators
        {
            'sma_short': Short-term SMA series,
            'sma_long': Long-term SMA series,
            'stable_direction': Stable direction {-1, 0, 1},
            'stable_strength': Stable strength [0.0-inf] (to be normalized later)
        }
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        if len(prices) < max(short_window, long_window):
            raise DataInsufficientError(f"Insufficient data: {len(prices)} < {max(short_window, long_window)}")
        
        # Calculate SMAs
        sma_short = prices.rolling(window=short_window).mean()
        sma_long = prices.rolling(window=long_window).mean()
        
        # Calculate stable direction
        sma_diff = sma_short - sma_long
        #stable_direction = np.sign(sma_diff).astype(int)
        stable_direction = np.sign(sma_diff).fillna(0).astype(int)

        # Calculate stable strength (relative to long-term SMA)
        stable_strength = np.abs(sma_diff) / sma_long
        
        # Handle potential division by zero
        stable_strength = stable_strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'stable_direction': stable_direction,
            'stable_strength': stable_strength
        }
        
    except Exception as e:
        logger.error(f"SMA trend calculation failed: {e}")
        raise DirectionConverterError(f"SMA calculation failed: {e}")


# ===========================================
# PARAMETER VALIDATION
# ===========================================

def validate_conversion_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and supplement conversion parameters.
    
    Args:
        params: Input parameters dictionary
        
    Returns:
        Validated parameters with default values filled
    """
    # Default parameters based on design document
    default_params = {
        # Window sizes
        'ema_short_window': 12,
        'ema_long_window': 26,
        'sma_short_window': 20,
        'sma_long_window': 50,
        'dfa_window': 100,
        'garch_vol_window': 252,
        
        # GARCH model parameters
        'garch_p': 1,
        'garch_q': 1,
        
        # Judgment parameters
        'volatility_k': 0.5,
        'hurst_trend_threshold': 0.55,
        'hurst_reversion_threshold': 0.45,
        
        # Pattern generation parameters
        'pattern_min_length': 3,
        'pattern_max_length': 10,
        'trend_min_duration': 2,
        'smoothing_window': 3,
    }
    
    # Merge with defaults
    if params is None:
        params = {}
    
    validated_params = default_params.copy()
    validated_params.update(params)
    
    # Validate parameter ranges
    errors = []
    
    # Window size validations
    if validated_params['ema_short_window'] >= validated_params['ema_long_window']:
        errors.append("ema_short_window must be < ema_long_window")
    
    if validated_params['sma_short_window'] >= validated_params['sma_long_window']:
        errors.append("sma_short_window must be < sma_long_window")
    
    if validated_params['dfa_window'] < 50:
        errors.append("dfa_window must be >= 50 for reliable Hurst calculation")
    
    if validated_params['garch_vol_window'] < 100:
        errors.append("garch_vol_window must be >= 100 for GARCH stability")
    
    # GARCH parameter validations
    if not (1 <= validated_params['garch_p'] <= 3):
        errors.append("garch_p must be in [1, 3] range")
    
    if not (1 <= validated_params['garch_q'] <= 3):
        errors.append("garch_q must be in [1, 3] range")
    
    # Threshold validations
    if not (0.0 < validated_params['volatility_k'] < 2.0):
        errors.append("volatility_k must be in (0.0, 2.0) range")
    
    if not (0.3 <= validated_params['hurst_trend_threshold'] <= 0.7):
        errors.append("hurst_trend_threshold must be in [0.3, 0.7] range")
    
    if not (0.3 <= validated_params['hurst_reversion_threshold'] <= 0.7):
        errors.append("hurst_reversion_threshold must be in [0.3, 0.7] range")
    
    if validated_params['hurst_reversion_threshold'] >= validated_params['hurst_trend_threshold']:
        errors.append("hurst_reversion_threshold must be < hurst_trend_threshold")
    
    # Pattern parameter validations
    if not (2 <= validated_params['pattern_min_length'] <= validated_params['pattern_max_length'] <= 15):
        errors.append("pattern lengths must satisfy: 2 <= min_length <= max_length <= 15")
    
    if validated_params['trend_min_duration'] < 1:
        errors.append("trend_min_duration must be >= 1")
    
    if validated_params['smoothing_window'] < 1:
        errors.append("smoothing_window must be >= 1")
    
    if errors:
        error_msg = "Parameter validation failed:\n" + "\n".join(f"- {err}" for err in errors)
        raise ValueError(error_msg)
    
    logger.info("Parameter validation passed")
    return validated_params


# ===========================================
# TASK 2: GARCH VOLATILITY PREDICTION
# ===========================================

def calculate_garch_volatility(returns: pd.Series, window: int, p: int, q: int) -> pd.Series:
    """
    Rolling GARCH(p,q) model for 1-step ahead volatility prediction.
    
    Algorithm:
    1. Apply rolling window of length 'window' to returns time series
    2. For each window, fit GARCH(p,q) model:
       model = arch_model(window_data, vol='Garch', p=p, q=q, rescale=False)
       results = model.fit(disp='off', show_warning=False)
    3. Forecast 1-step ahead conditional volatility:
       forecast = results.forecast(horizon=1)
       predicted_vol = sqrt(forecast.variance.iloc[-1, 0])
    4. Exception handling for non-convergent models:
       fallback = window_data.std() * sqrt(252)  # Annualized historical volatility
    5. Accumulate predicted volatilities as time series
    
    Args:
        returns: Daily returns series
        window: Rolling window size for GARCH fitting
        p: GARCH(p,q) parameter p
        q: GARCH(p,q) parameter q
        
    Returns:
        predicted_volatility_series: 1-step ahead predicted volatility time series
        
    Notes:
        - First 'window' periods are filled with historical volatility
        - For computational efficiency, consider weekly updates (every 5 days)
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        if not ARCH_AVAILABLE:
            logger.warning("arch package not available, using historical volatility fallback")
            return _fallback_historical_volatility_series(returns, window)
        
        if len(returns) < window:
            raise DataInsufficientError(f"Insufficient data: {len(returns)} < {window}")
        
        # Initialize result series with NaN
        predicted_vol = pd.Series(index=returns.index, dtype=float)
        
        # Fill initial period with historical volatility
        for i in range(window):
            if i < 30:  # Need minimum 30 days for meaningful volatility
                predicted_vol.iloc[i] = np.nan
            else:
                historical_vol = returns.iloc[:i+1].std() * np.sqrt(252)
                predicted_vol.iloc[i] = historical_vol
        
        # Rolling GARCH prediction
        for i in range(window, len(returns)):
            window_data = returns.iloc[i-window:i]
            
            # Skip if too many NaN values in window
            if window_data.isnull().sum() > window * 0.1:  # Allow up to 10% missing
                predicted_vol.iloc[i] = predicted_vol.iloc[i-1]  # Use previous value
                continue
            
            # Clean data for GARCH
            clean_data = window_data.dropna()
            if len(clean_data) < window * 0.8:  # Need at least 80% of window
                predicted_vol.iloc[i] = _fallback_historical_volatility(clean_data)
                continue
            
            try:
                vol_prediction = _fit_garch_model(clean_data, p, q)
                predicted_vol.iloc[i] = vol_prediction
            except (GARCHConvergenceError, Exception) as e:
                logger.debug(f"GARCH failed at {returns.index[i]}: {e}")
                # Use fallback
                predicted_vol.iloc[i] = _fallback_historical_volatility(clean_data)
        
        # Forward fill any remaining NaN values
        predicted_vol = predicted_vol.fillna(method='ffill')
        
        # Final fallback if all NaN
        if predicted_vol.isnull().all():
            logger.warning("All GARCH predictions failed, using constant historical volatility")
            predicted_vol = predicted_vol.fillna(returns.std() * np.sqrt(252))
        
        logger.info(f"GARCH volatility prediction completed: {(~predicted_vol.isnull()).sum()}/{len(predicted_vol)} valid predictions")
        return predicted_vol
        
    except Exception as e:
        logger.error(f"GARCH volatility calculation failed: {e}")
        raise DirectionConverterError(f"GARCH calculation failed: {e}")


def _fit_garch_model(window_data: pd.Series, p: int, q: int) -> float:
    """
    Fit GARCH model to single window and predict 1-step ahead volatility.
    
    Args:
        window_data: Returns data for the window (clean, no NaN)
        p: GARCH(p,q) parameter p
        q: GARCH(p,q) parameter q
        
    Returns:
        predicted_volatility: 1-step ahead predicted volatility (annualized)
        
    Raises:
        GARCHConvergenceError: If model fails to converge
    """
    if not ARCH_AVAILABLE:
        raise GARCHConvergenceError("arch package not available")
    
    try:
        # Convert to percentage returns for better numerical stability
        returns_pct = window_data * 100
        
        # Check for sufficient variation
        if returns_pct.std() < 1e-8:
            raise GARCHConvergenceError("Insufficient variation in returns")
        
        # Create GARCH model
        model = arch_model(
            returns_pct, 
            vol='Garch', 
            p=p, 
            q=q, 
            rescale=False,
            mean='Zero'  # Assume zero mean for returns
        )
        
        # Fit model with suppressed output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.fit(
                disp='off', 
                show_warning=False,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
        
        # Check convergence
        if not results.convergence_flag:
            raise GARCHConvergenceError("GARCH optimization did not converge")
        
        # Forecast 1-step ahead
        forecast = results.forecast(horizon=1, method='simulation', simulations=1000)
        
        # Extract predicted variance and convert to volatility
        predicted_variance = forecast.variance.iloc[-1, 0]
        predicted_vol_daily = np.sqrt(predicted_variance) / 100  # Convert back from percentage
        
        # Annualize (252 trading days)
        predicted_vol_annual = predicted_vol_daily * np.sqrt(252)
        
        # Sanity check: volatility should be reasonable
        if not (0.01 <= predicted_vol_annual <= 5.0):  # 1% to 500% annual volatility
            raise GARCHConvergenceError(f"Unreasonable volatility prediction: {predicted_vol_annual:.3f}")
        
        return predicted_vol_annual
        
    except Exception as e:
        if isinstance(e, GARCHConvergenceError):
            raise
        else:
            raise GARCHConvergenceError(f"GARCH fitting failed: {e}")


def _fallback_historical_volatility(window_data: pd.Series) -> float:
    """
    Calculate fallback historical volatility when GARCH fails.
    
    Args:
        window_data: Returns data for the window
        
    Returns:
        historical_volatility: Annualized historical volatility
    """
    try:
        if len(window_data) < 10:  # Need minimum data
            return 0.2  # Default 20% annual volatility
        
        # Remove outliers (beyond 3 standard deviations)
        clean_data = window_data.copy()
        std_threshold = 3 * clean_data.std()
        clean_data = clean_data[np.abs(clean_data - clean_data.mean()) <= std_threshold]
        
        if len(clean_data) < 5:
            return 0.2  # Default if too much data removed
        
        # Calculate annualized historical volatility
        daily_vol = clean_data.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Ensure reasonable range
        annual_vol = np.clip(annual_vol, 0.05, 3.0)  # 5% to 300% annual volatility
        
        return annual_vol
        
    except Exception:
        return 0.2  # Ultimate fallback


def _fallback_historical_volatility_series(returns: pd.Series, window: int) -> pd.Series:
    """
    Create entire volatility series using rolling historical volatility.
    
    Args:
        returns: Daily returns series
        window: Rolling window size
        
    Returns:
        volatility_series: Rolling historical volatility series
    """
    try:
        # Calculate rolling standard deviation
        rolling_vol = returns.rolling(window=window, min_periods=10).std()
        
        # Annualize
        annual_vol = rolling_vol * np.sqrt(252)
        
        # Fill initial NaN values
        annual_vol = annual_vol.fillna(method='bfill')
        
        # Ensure reasonable range
        annual_vol = annual_vol.clip(0.05, 3.0)
        
        return annual_vol
        
    except Exception as e:
        logger.error(f"Fallback volatility calculation failed: {e}")
        # Return constant volatility as last resort
        return pd.Series(0.2, index=returns.index)


# ===========================================
# TASK 3: HURST EXPONENT CALCULATION (DFA)
# ===========================================

def calculate_hurst_exponent(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).
    
    Algorithm:
    1. log_prices = log(prices)
    2. cumsum_series = (log_prices - log_prices.mean()).cumsum()
    3. For each scale n in [4, 8, 16, 32, 64]:
        a. Divide series into segments of length n
        b. Detrend each segment (linear regression)
        c. Calculate fluctuation F(n) = sqrt(mean(residuals^2))
    4. Hurst exponent = slope of log(n) vs log(F(n)) linear regression
    5. Calculate as rolling window time series
    
    Args:
        prices: Price series
        window: Rolling window size for Hurst calculation
        
    Returns:
        hurst_series: Rolling Hurst exponent time series [0.0-1.0]
        
    Interpretation:
        H > 0.5: Trend persistence (long memory)
        H < 0.5: Mean reversion tendency
        H ≈ 0.5: Random walk
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        if not SCIPY_SKLEARN_AVAILABLE:
            logger.warning("scipy/sklearn not available, using simplified Hurst calculation")
            return _simplified_hurst_calculation(prices, window)
        
        if len(prices) < window:
            raise DataInsufficientError(f"Insufficient data: {len(prices)} < {window}")
        
        if window < 100:
            logger.warning(f"Window size {window} may be too small for reliable Hurst calculation")
        
        # Initialize result series
        hurst_series = pd.Series(index=prices.index, dtype=float)
        
        # Calculate rolling Hurst exponent
        for i in range(window, len(prices)):
            window_prices = prices.iloc[i-window:i]
            
            # Skip if too many NaN values
            if window_prices.isnull().sum() > window * 0.1:
                hurst_series.iloc[i] = hurst_series.iloc[i-1] if i > window else 0.5
                continue
            
            # Clean data
            clean_prices = window_prices.dropna()
            if len(clean_prices) < window * 0.8:
                hurst_series.iloc[i] = 0.5  # Default to random walk
                continue
            
            try:
                hurst_value = _dfa_single_window(clean_prices.values)
                hurst_series.iloc[i] = hurst_value
            except Exception as e:
                logger.debug(f"Hurst calculation failed at {prices.index[i]}: {e}")
                hurst_series.iloc[i] = 0.5  # Default to random walk
        
        # Fill initial values with 0.5 (random walk assumption)
        hurst_series.iloc[:window] = 0.5
        
        # Forward fill any remaining NaN values
        hurst_series = hurst_series.fillna(method='ffill')
        
        # Ensure values are in valid range [0.0, 1.0]
        hurst_series = hurst_series.clip(0.0, 1.0)
        
        logger.info(f"Hurst exponent calculation completed: {(~hurst_series.isnull()).sum()}/{len(hurst_series)} valid values")
        return hurst_series
        
    except Exception as e:
        logger.error(f"Hurst exponent calculation failed: {e}")
        raise DirectionConverterError(f"Hurst calculation failed: {e}")


def _dfa_single_window(price_data, scales: List[int] = None) -> float:
    """
    Perform DFA calculation for single window.
    
    Args:
        price_data: Price data array (clean, no NaN)
        scales: List of scales for DFA analysis
        
    Returns:
        hurst_exponent: Hurst exponent value [0.0-1.0]
    """
    if scales is None:
        scales = [4, 8, 16, 32, 64]
    
    # Filter scales based on data length
    max_scale = len(price_data) // 4  # Need at least 4 segments
    valid_scales = [s for s in scales if s <= max_scale]
    
    if len(valid_scales) < 3:
        # Not enough scales for reliable calculation
        return 0.5
    
    try:
        # Step 1: Convert to log prices and create cumulative sum
        log_prices = np.log(price_data)
        log_prices_centered = log_prices - np.mean(log_prices)
        cumsum_series = np.cumsum(log_prices_centered)
        
        # Step 2: Calculate fluctuations for each scale
        fluctuations = []
        log_scales = []
        
        for scale in valid_scales:
            fluctuation = _calculate_fluctuation(cumsum_series, scale)
            if fluctuation > 0:  # Valid fluctuation
                fluctuations.append(fluctuation)
                log_scales.append(np.log(scale))
        
        if len(fluctuations) < 3:
            return 0.5
        
        # Step 3: Linear regression to find Hurst exponent
        log_fluctuations = np.log(fluctuations)
        
        # Use sklearn for robust linear regression
        X = np.array(log_scales).reshape(-1, 1)
        y = np.array(log_fluctuations)
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        hurst_exponent = reg.coef_[0]
        
        # Ensure reasonable range
        hurst_exponent = np.clip(hurst_exponent, 0.1, 0.9)
        
        return hurst_exponent
        
    except Exception as e:
        logger.debug(f"DFA single window calculation failed: {e}")
        return 0.5


def _calculate_fluctuation(cumsum_data, scale: int) -> float:
    """
    Calculate fluctuation for a given scale in DFA.
    
    Args:
        cumsum_data: Cumulative sum data
        scale: Scale (segment length)
        
    Returns:
        fluctuation: RMS fluctuation for the scale
    """
    try:
        n = len(cumsum_data)
        
        # Number of complete segments
        num_segments = n // scale
        
        if num_segments < 2:
            return 0.0
        
        fluctuations = []
        
        # Process each segment
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = (i + 1) * scale
            segment = cumsum_data[start_idx:end_idx]
            
            # Detrend segment using linear regression
            rms_residual = _detrend_segment(segment)
            if rms_residual > 0:
                fluctuations.append(rms_residual)
        
        if len(fluctuations) == 0:
            return 0.0
        
        # Return root mean square of fluctuations
        return np.sqrt(np.mean(np.array(fluctuations) ** 2))
        
    except Exception:
        return 0.0


def _detrend_segment(segment) -> float:
    """
    Detrend segment using linear regression and return RMS residual.
    
    Args:
        segment: Data segment to detrend
        
    Returns:
        rms_residual: Root mean square of residuals
    """
    try:
        n = len(segment)
        if n < 3:
            return 0.0
        
        # Create time index
        t = np.arange(n)
        
        # Linear regression
        X = np.column_stack([np.ones(n), t])
        
        # Use least squares to find linear trend
        coeffs, residuals, rank, s = np.linalg.lstsq(X, segment, rcond=None)
        
        # Calculate fitted values and residuals
        fitted = X @ coeffs
        residuals = segment - fitted
        
        # Return RMS of residuals
        return np.sqrt(np.mean(residuals ** 2))
        
    except Exception:
        return 0.0


def _simplified_hurst_calculation(prices: pd.Series, window: int) -> pd.Series:
    """
    Simplified Hurst calculation when scipy/sklearn not available.
    
    Uses rescaled range (R/S) method as approximation.
    """
    try:
        hurst_series = pd.Series(index=prices.index, dtype=float)
        
        for i in range(window, len(prices)):
            window_prices = prices.iloc[i-window:i]
            clean_prices = window_prices.dropna()
            
            if len(clean_prices) < window * 0.8:
                hurst_series.iloc[i] = 0.5
                continue
            
            # Simple R/S calculation
            returns = clean_prices.pct_change().dropna()
            if len(returns) < 10:
                hurst_series.iloc[i] = 0.5
                continue
            
            # Calculate cumulative deviation from mean
            mean_return = returns.mean()
            cumdev = (returns - mean_return).cumsum()
            
            # Calculate range and standard deviation
            R = cumdev.max() - cumdev.min()
            S = returns.std()
            
            if S > 0:
                rs_ratio = R / S
                # Approximate Hurst using R/S relationship
                hurst_approx = np.log(rs_ratio) / np.log(len(returns))
                hurst_series.iloc[i] = np.clip(hurst_approx, 0.1, 0.9)
            else:
                hurst_series.iloc[i] = 0.5
        
        # Fill initial values
        hurst_series.iloc[:window] = 0.5
        hurst_series = hurst_series.fillna(0.5)
        
        return hurst_series
        
    except Exception as e:
        logger.error(f"Simplified Hurst calculation failed: {e}")
        return pd.Series(0.5, index=prices.index)


# ===========================================
# TASK 4: INTEGRATED DIRECTION DETERMINATION
# ===========================================

def determine_final_direction(
    price_returns: pd.Series,
    ema_data: Dict[str, pd.Series],
    sma_data: Dict[str, pd.Series],
    predicted_volatility: pd.Series,
    hurst: pd.Series,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Multi-indicator final direction determination using GARCH predicted volatility.
    
    Algorithm:
    1. Calculate dynamic threshold:
       daily_vol_threshold = predicted_volatility / sqrt(252)
       dynamic_threshold = params['volatility_k'] * daily_vol_threshold
    
    2. For each timestamp, determine:
        a. Sideways judgment:
           if abs(price_returns) < dynamic_threshold:
               direction = 0
        
        b. Directional judgment:
           else:
               ema_vote = ema_data['trend_direction']
               sma_vote = sma_data['stable_direction']
               
               # Hurst-based weight adjustment
               if hurst > params['hurst_trend_threshold']:      # High trend persistence
                   weight_ema = 0.7
               elif hurst < params['hurst_reversion_threshold']: # High mean reversion
                   weight_ema = 0.3
               else:                                            # Intermediate
                   weight_ema = 0.5
               
               final_vote = weight_ema * ema_vote + (1 - weight_ema) * sma_vote
               direction = sign(final_vote).astype(int)
    
    3. Strength calculation:
       strength = weight_ema * ema_data['trend_strength'] + (1 - weight_ema) * sma_data['stable_strength']
       strength_normalized = (strength - strength.min()) / (strength.max() - strength.min())
    
    Args:
        price_returns: Daily returns series
        ema_data: EMA trend data dictionary
        sma_data: SMA trend data dictionary
        predicted_volatility: GARCH predicted volatility (annualized)
        hurst: Hurst exponent series
        params: Conversion parameters
        
    Returns:
        DataFrame with columns: ['direction', 'strength', 'volatility', 'hurst']
        - direction: {-1, 0, 1}
        - strength: [0.0-1.0]
        - volatility: [0.0-inf] (daily volatility)
        - hurst: [0.0-1.0]
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        # Ensure all inputs have the same index
        common_index = price_returns.index.intersection(predicted_volatility.index)
        common_index = common_index.intersection(hurst.index)
        
        if len(common_index) == 0:
            raise DirectionConverterError("No common dates between input series")
        
        # Align all data to common index
        aligned_returns = price_returns.reindex(common_index)
        aligned_volatility = predicted_volatility.reindex(common_index)
        aligned_hurst = hurst.reindex(common_index)
        
        # Align EMA/SMA data
        aligned_ema_direction = ema_data['trend_direction'].reindex(common_index)
        aligned_ema_strength = ema_data['trend_strength'].reindex(common_index)
        aligned_sma_direction = sma_data['stable_direction'].reindex(common_index)
        aligned_sma_strength = sma_data['stable_strength'].reindex(common_index)
        
        # Step 1: Calculate dynamic threshold
        # Convert annual volatility to daily
        daily_volatility = aligned_volatility / np.sqrt(252)
        dynamic_threshold = params['volatility_k'] * daily_volatility
        
        # Step 2: Determine direction for each timestamp
        directions = []
        strengths = []
        
        for i in range(len(common_index)):
            current_return = aligned_returns.iloc[i]
            current_threshold = dynamic_threshold.iloc[i]
            current_hurst = aligned_hurst.iloc[i]
            
            # Handle NaN values
            if pd.isna(current_return) or pd.isna(current_threshold) or pd.isna(current_hurst):
                directions.append(0)
                strengths.append(0.0)
                continue
            
            # Sideways judgment
            if abs(current_return) < current_threshold:
                direction = 0
                strength = 0.0
            else:
                # Get EMA and SMA votes
                ema_vote = aligned_ema_direction.iloc[i] if not pd.isna(aligned_ema_direction.iloc[i]) else 0
                sma_vote = aligned_sma_direction.iloc[i] if not pd.isna(aligned_sma_direction.iloc[i]) else 0
                
                # Calculate Hurst-based weight
                weight_ema = _calculate_hurst_weight(current_hurst, params)
                
                # Final weighted vote
                final_vote = weight_ema * ema_vote + (1 - weight_ema) * sma_vote
                direction = int(np.sign(final_vote))
                
                # Calculate strength
                ema_str = aligned_ema_strength.iloc[i] if not pd.isna(aligned_ema_strength.iloc[i]) else 0.0
                sma_str = aligned_sma_strength.iloc[i] if not pd.isna(aligned_sma_strength.iloc[i]) else 0.0
                strength = weight_ema * ema_str + (1 - weight_ema) * sma_str
            
            directions.append(direction)
            strengths.append(strength)
        
        # Step 3: Normalize strength values
        strength_series = pd.Series(strengths, index=common_index)
        normalized_strength = _normalize_strength_values(strength_series)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'direction': directions,
            'strength': normalized_strength,
            'volatility': daily_volatility,  # Daily volatility
            'hurst': aligned_hurst
        }, index=common_index)
        
        # Final validation
        result_df['direction'] = result_df['direction'].astype(int)
        result_df['strength'] = result_df['strength'].clip(0.0, 1.0)
        result_df['volatility'] = result_df['volatility'].clip(0.0, np.inf)
        result_df['hurst'] = result_df['hurst'].clip(0.0, 1.0)
        
        logger.info(f"Final direction determination completed: {len(result_df)} data points")
        return result_df
        
    except Exception as e:
        logger.error(f"Final direction determination failed: {e}")
        raise DirectionConverterError(f"Direction determination failed: {e}")


def _calculate_hurst_weight(hurst_value: float, params: Dict[str, Any]) -> float:
    """
    Calculate Hurst-based weight for EMA vs SMA combination.
    
    Args:
        hurst_value: Current Hurst exponent value
        params: Parameters containing thresholds
        
    Returns:
        weight_ema: Weight for EMA [0.0-1.0]
    """
    hurst_trend_threshold = params['hurst_trend_threshold']
    hurst_reversion_threshold = params['hurst_reversion_threshold']
    
    if hurst_value > hurst_trend_threshold:
        # High trend persistence - favor EMA (faster reaction)
        weight_ema = 0.7
    elif hurst_value < hurst_reversion_threshold:
        # High mean reversion - favor SMA (stability)
        weight_ema = 0.3
    else:
        # Intermediate - balanced weight
        weight_ema = 0.5
    
    return weight_ema


def _normalize_strength_values(strength_series: pd.Series) -> pd.Series:
    """
    Normalize strength values to [0.0-1.0] range with outlier handling.
    
    Args:
        strength_series: Raw strength values
        
    Returns:
        normalized_strength: Normalized strength [0.0-1.0]
    """
    try:
        # Handle outliers by clipping at 99th percentile
        upper_bound = strength_series.quantile(0.99)
        clipped_strength = strength_series.clip(0, upper_bound)
        
        # Min-max normalization
        min_val = clipped_strength.min()
        max_val = clipped_strength.max()
        
        if max_val == min_val:
            # Constant series - return midpoint
            return pd.Series(0.5, index=strength_series.index)
        
        normalized = (clipped_strength - min_val) / (max_val - min_val)
        
        # Ensure values are in [0.0, 1.0]
        normalized = normalized.clip(0.0, 1.0)
        
        return normalized
        
    except Exception as e:
        logger.error(f"Strength normalization failed: {e}")
        return pd.Series(0.5, index=strength_series.index)


def _apply_dynamic_threshold(returns: pd.Series, volatility: pd.Series, k: float) -> pd.Series:
    """
    Apply dynamic threshold based on predicted volatility.
    
    Args:
        returns: Daily returns
        volatility: Predicted volatility (annualized)
        k: Volatility coefficient
        
    Returns:
        threshold_series: Dynamic threshold series
    """
    try:
        # Convert to daily volatility
        daily_vol = volatility / np.sqrt(252)
        
        # Apply coefficient
        threshold = k * daily_vol
        
        # Ensure reasonable bounds
        threshold = threshold.clip(0.001, 0.1)  # 0.1% to 10% daily threshold
        
        return threshold
        
    except Exception as e:
        logger.error(f"Dynamic threshold calculation failed: {e}")
        return pd.Series(0.02, index=returns.index)  # 2% default


# ===========================================
# TASK 5: PATTERN GENERATION & TREND DURATION
# ===========================================

def calculate_trend_duration(directions: pd.Series) -> pd.Series:
    """
    Calculate consecutive same-direction trend duration.
    
    Algorithm:
    1. direction_changes = (directions != directions.shift(1))
    2. trend_groups = direction_changes.cumsum()
    3. For each group:
        duration = group.size
        assign duration to all members of the group
    
    Args:
        directions: Direction series {-1, 0, 1}
        
    Returns:
        trend_duration: Consecutive days count for each timestamp
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        if len(directions) == 0:
            return pd.Series(dtype=int)
        
        # Identify direction changes
        direction_changes = (directions != directions.shift(1))
        
        # Create group identifiers
        trend_groups = direction_changes.cumsum()
        
        # Calculate duration for each group
        group_sizes = directions.groupby(trend_groups).size()
        
        # Map durations back to original index
        trend_duration = directions.map(lambda x: 1)  # Initialize
        
        for group_id in trend_groups.unique():
            if pd.isna(group_id):
                continue
            group_mask = (trend_groups == group_id)
            group_size = group_sizes.loc[group_id]
            trend_duration.loc[group_mask] = group_size
        
        # Ensure integer type
        trend_duration = trend_duration.astype(int)
        
        logger.info(f"Trend duration calculation completed: max duration = {trend_duration.max()}")
        return trend_duration
        
    except Exception as e:
        logger.error(f"Trend duration calculation failed: {e}")
        raise DirectionConverterError(f"Trend duration calculation failed: {e}")


def generate_pattern_sequences(
    directions_df: pd.DataFrame,
    min_length: int,
    max_length: int
) -> pd.DataFrame:
    """
    Generate higher-order patterns from consecutive direction combinations.
    
    Algorithm:
    1. For pattern_length in range(min_length, max_length + 1):
        2. For each possible starting position:
            a. Extract sequence of 'direction' values
            b. Convert to pattern_code (e.g., "110", "-1-10", "001")
            c. Calculate pattern_strength = mean(strength values in sequence)
            d. Store with start_date and end_date
    3. Filter overlapping patterns (keep strongest)
    4. Ensure minimum quality threshold (pattern_strength >= 0.3)
    
    Args:
        directions_df: DataFrame with columns ['direction', 'strength', ...]
        min_length: Minimum pattern length
        max_length: Maximum pattern length
        
    Returns:
        DataFrame with columns: ['pattern_length', 'pattern_code', 'pattern_strength', 'start_date']
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        if len(directions_df) == 0:
            return pd.DataFrame(columns=['pattern_length', 'pattern_code', 'pattern_strength', 'start_date'])
        
        required_columns = ['direction', 'strength']
        if not all(col in directions_df.columns for col in required_columns):
            raise DirectionConverterError(f"Missing required columns: {required_columns}")
        
        patterns = []
        
        # Generate patterns for each length
        for pattern_length in range(min_length, max_length + 1):
            for start_idx in range(len(directions_df) - pattern_length + 1):
                pattern_data = _extract_pattern_sequence(
                    directions_df, start_idx, pattern_length
                )
                
                if pattern_data['pattern_strength'] >= 0.3:  # Quality threshold
                    patterns.append(pattern_data)
        
        if len(patterns) == 0:
            logger.warning("No patterns found meeting quality threshold")
            return pd.DataFrame(columns=['pattern_length', 'pattern_code', 'pattern_strength', 'start_date'])
        
        # Convert to DataFrame
        patterns_df = pd.DataFrame(patterns)
        
        # Filter overlapping patterns (keep strongest)
        filtered_patterns = _filter_overlapping_patterns(patterns)
        
        if len(filtered_patterns) == 0:
            logger.warning("No patterns remaining after overlap filtering")
            return pd.DataFrame(columns=['pattern_length', 'pattern_code', 'pattern_strength', 'start_date'])
        
        # Create final DataFrame
        result_df = pd.DataFrame(filtered_patterns)
        
        # Set index to end_date for consistency with schema
        if 'end_date' in result_df.columns:
            result_df = result_df.set_index('end_date')
        
        logger.info(f"Pattern generation completed: {len(result_df)} patterns found")
        return result_df
        
    except Exception as e:
        logger.error(f"Pattern sequence generation failed: {e}")
        raise DirectionConverterError(f"Pattern generation failed: {e}")


def _extract_pattern_sequence(directions_df: pd.DataFrame, start_idx: int, length: int) -> Dict[str, Any]:
    """
    Extract pattern sequence from specified position and length.
    
    Args:
        directions_df: DataFrame with direction and strength data
        start_idx: Starting index
        length: Pattern length
        
    Returns:
        pattern_data: Dictionary with pattern information
    """
    try:
        # Extract sequence
        end_idx = start_idx + length
        sequence = directions_df.iloc[start_idx:end_idx]
        
        # Create pattern code
        directions = sequence['direction'].values
        pattern_code = ''.join(str(int(d)) for d in directions)
        
        # Calculate pattern strength
        pattern_strength = _calculate_pattern_strength(sequence['strength'])
        
        # Get dates
        start_date = sequence.index[0]
        end_date = sequence.index[-1]
        
        return {
            'pattern_length': length,
            'pattern_code': pattern_code,
            'pattern_strength': pattern_strength,
            'start_date': start_date,
            'end_date': end_date
        }
        
    except Exception as e:
        logger.debug(f"Pattern extraction failed at {start_idx}: {e}")
        return {
            'pattern_length': length,
            'pattern_code': '',
            'pattern_strength': 0.0,
            'start_date': None,
            'end_date': None
        }


def _filter_overlapping_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter overlapping patterns, keeping the strongest ones.
    
    Args:
        patterns: List of pattern dictionaries
        
    Returns:
        filtered_patterns: Non-overlapping patterns
    """
    try:
        if len(patterns) <= 1:
            return patterns
        
        # Sort by strength (descending)
        sorted_patterns = sorted(patterns, key=lambda x: x['pattern_strength'], reverse=True)
        
        filtered = []
        used_dates = set()
        
        for pattern in sorted_patterns:
            start_date = pattern.get('start_date')
            end_date = pattern.get('end_date')
            
            if start_date is None or end_date is None:
                continue
            
            # Create date range for this pattern
            pattern_dates = pd.date_range(start_date, end_date, freq='D')
            
            # Check for overlap with already selected patterns
            if not any(date in used_dates for date in pattern_dates):
                filtered.append(pattern)
                used_dates.update(pattern_dates)
        
        return filtered
        
    except Exception as e:
        logger.error(f"Pattern overlap filtering failed: {e}")
        return patterns


def _calculate_pattern_strength(strength_sequence: pd.Series) -> float:
    """
    Calculate pattern strength from strength sequence.
    
    Args:
        strength_sequence: Series of strength values
        
    Returns:
        pattern_strength: Aggregate pattern strength [0.0-1.0]
    """
    try:
        if len(strength_sequence) == 0:
            return 0.0
        
        # Remove NaN values
        clean_strength = strength_sequence.dropna()
        
        if len(clean_strength) == 0:
            return 0.0
        
        # Use mean with consistency bonus
        mean_strength = clean_strength.mean()
        
        # Consistency bonus: lower standard deviation = higher bonus
        if len(clean_strength) > 1:
            consistency = 1.0 - (clean_strength.std() / (clean_strength.mean() + 1e-8))
            consistency = max(0.0, min(1.0, consistency))
            
            # Weighted combination
            pattern_strength = 0.7 * mean_strength + 0.3 * consistency
        else:
            pattern_strength = mean_strength
        
        return max(0.0, min(1.0, pattern_strength))
        
    except Exception:
        return 0.0


# ===========================================
# TASK 6: QUALITY EVALUATION & MAIN FUNCTIONS
# ===========================================

def calculate_quality_metrics(
    directions_df: pd.DataFrame,
    pattern_sequences: pd.DataFrame,
    original_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate conversion quality indicators.
    
    Quality Metrics:
    1. pattern_coverage = len(pattern_sequences) / len(directions_df)
    2. avg_pattern_strength = pattern_sequences['pattern_strength'].mean()
    3. data_completeness = (1 - directions_df.isnull().sum().sum() / directions_df.size)
    4. direction_consistency = correlation between consecutive directions
    5. volatility_adaptation = correlation between predicted volatility and actual returns volatility
    
    Args:
        directions_df: Direction analysis results
        pattern_sequences: Pattern sequences
        original_data: Original price data for reference
        
    Returns:
        quality_metrics: Dict[str, float] with all metrics [0.0-1.0]
    """
    try:
        metrics = {}
        
        # 1. Pattern coverage
        if len(directions_df) > 0:
            metrics['pattern_coverage'] = len(pattern_sequences) / len(directions_df)
        else:
            metrics['pattern_coverage'] = 0.0
        
        # 2. Average pattern strength
        if len(pattern_sequences) > 0 and 'pattern_strength' in pattern_sequences.columns:
            metrics['avg_pattern_strength'] = pattern_sequences['pattern_strength'].mean()
        else:
            metrics['avg_pattern_strength'] = 0.0
        
        # 3. Data completeness
        total_cells = directions_df.size
        missing_cells = directions_df.isnull().sum().sum()
        metrics['data_completeness'] = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
        
        # 4. Direction consistency
        if 'direction' in directions_df.columns and len(directions_df) > 1:
            directions = directions_df['direction'].dropna()
            if len(directions) > 1:
                direction_changes = (directions != directions.shift(1)).sum()
                total_periods = len(directions) - 1
                consistency = 1.0 - (direction_changes / total_periods) if total_periods > 0 else 0.0
                metrics['direction_consistency'] = max(0.0, min(1.0, consistency))
            else:
                metrics['direction_consistency'] = 1.0
        else:
            metrics['direction_consistency'] = 0.0
        
        # 5. Volatility adaptation
        if 'volatility' in directions_df.columns and 'close' in original_data.columns:
            try:
                predicted_vol = directions_df['volatility'].dropna()
                actual_returns = original_data['close'].pct_change().dropna()
                
                if len(predicted_vol) > 10 and len(actual_returns) > 10:
                    # Calculate rolling actual volatility
                    actual_vol = actual_returns.rolling(window=20).std() * np.sqrt(252)
                    
                    # Align data
                    common_index = predicted_vol.index.intersection(actual_vol.index)
                    if len(common_index) > 10:
                        pred_aligned = predicted_vol.reindex(common_index)
                        actual_aligned = actual_vol.reindex(common_index)
                        
                        correlation = pred_aligned.corr(actual_aligned)
                        metrics['volatility_adaptation'] = max(0.0, correlation) if not pd.isna(correlation) else 0.0
                    else:
                        metrics['volatility_adaptation'] = 0.0
                else:
                    metrics['volatility_adaptation'] = 0.0
            except Exception:
                metrics['volatility_adaptation'] = 0.0
        else:
            metrics['volatility_adaptation'] = 0.0
        
        # Ensure all metrics are in [0.0, 1.0] range
        for key, value in metrics.items():
            metrics[key] = max(0.0, min(1.0, value))
        
        logger.info(f"Quality metrics calculated: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Quality metrics calculation failed: {e}")
        return {
            'pattern_coverage': 0.0,
            'avg_pattern_strength': 0.0,
            'data_completeness': 0.0,
            'direction_consistency': 0.0,
            'volatility_adaptation': 0.0
        }


def convert_to_direction_patterns(
    data: ProcessedDataContainer,
    conversion_params: Dict[str, Any] = None
) -> DirectionPatterns:
    """
    Main conversion function: orchestrate all processing steps.
    
    Pipeline:
    1. Parameter validation and default value setting
    2. Process BTC and MSTR data separately
    3. Calculate quality metrics
    4. Generate final DirectionPatterns object
    
    Args:
        data: Processed data container
        conversion_params: Conversion parameters
        
    Returns:
        DirectionPatterns: Complete direction pattern analysis results
    """
    try:
        if not PANDAS_AVAILABLE:
            raise DirectionConverterError("pandas not available")
        
        # Step 1: Validate parameters
        params = validate_conversion_params(conversion_params)
        logger.info("Starting direction pattern conversion")
        
        # Step 2: Process each asset
        btc_directions, btc_patterns = _process_single_asset(data.btc_processed, params, 'BTC')
        mstr_directions, mstr_patterns = _process_single_asset(data.mstr_processed, params, 'MSTR')
        
        # Step 3: Calculate quality metrics
        quality_metrics = {}
        
        # BTC quality metrics
        btc_quality = calculate_quality_metrics(btc_directions, btc_patterns, data.btc_processed)
        for key, value in btc_quality.items():
            quality_metrics[f'btc_{key}'] = value
        
        # MSTR quality metrics
        mstr_quality = calculate_quality_metrics(mstr_directions, mstr_patterns, data.mstr_processed)
        for key, value in mstr_quality.items():
            quality_metrics[f'mstr_{key}'] = value
        
        # Overall quality metrics
        quality_metrics['pattern_coverage'] = (quality_metrics['btc_pattern_coverage'] + quality_metrics['mstr_pattern_coverage']) / 2
        quality_metrics['avg_pattern_strength'] = (quality_metrics['btc_avg_pattern_strength'] + quality_metrics['mstr_avg_pattern_strength']) / 2
        quality_metrics['data_completeness'] = (quality_metrics['btc_data_completeness'] + quality_metrics['mstr_data_completeness']) / 2
        quality_metrics['direction_consistency'] = (quality_metrics['btc_direction_consistency'] + quality_metrics['mstr_direction_consistency']) / 2
        quality_metrics['volatility_adaptation'] = (quality_metrics['btc_volatility_adaptation'] + quality_metrics['mstr_volatility_adaptation']) / 2
        
        # Step 4: Create final result
        result = DirectionPatterns(
            btc_directions=btc_directions,
            mstr_directions=mstr_directions,
            btc_pattern_sequences=btc_patterns,
            mstr_pattern_sequences=mstr_patterns,
            conversion_params=params,
            quality_metrics=quality_metrics
        )
        
        # Step 5: Validate result
        if not result.validate():
            logger.warning("Direction patterns validation failed")
        
        logger.info("Direction pattern conversion completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Direction pattern conversion failed: {e}")
        raise DirectionConverterError(f"Conversion failed: {e}")


def _process_single_asset(asset_data: pd.DataFrame, params: Dict[str, Any], asset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process single asset through complete pipeline.
    
    Pipeline:
    1. Extract prices and returns
    2. Calculate each feature (EMA, SMA, GARCH, Hurst)
    3. Integrate all features to determine final direction
    4. Calculate trend duration and add to DataFrame
    5. Generate higher-order pattern sequences
    
    Args:
        asset_data: Asset DataFrame with OHLCV data
        params: Conversion parameters
        asset_name: Asset name for logging
        
    Returns:
        directions_df: Direction DataFrame
        pattern_sequences: Pattern sequences DataFrame
    """
    try:
        logger.info(f"Processing {asset_name} data")
        
        # Extract data
        if 'close' not in asset_data.columns:
            raise DirectionConverterError(f"{asset_name}: Missing 'close' column")
        
        prices = asset_data['close']
        returns = prices.pct_change().dropna()
        
        if len(prices) < 100:
            logger.warning(f"{asset_name}: Insufficient data ({len(prices)} points)")
        
        # Calculate features
        logger.debug(f"{asset_name}: Calculating EMA trend")
        ema_data = calculate_ema_trend(prices, params['ema_short_window'], params['ema_long_window'])
        
        logger.debug(f"{asset_name}: Calculating SMA trend")
        sma_data = calculate_sma_trend(prices, params['sma_short_window'], params['sma_long_window'])
        
        logger.debug(f"{asset_name}: Calculating GARCH volatility")
        predicted_volatility = calculate_garch_volatility(returns, params['garch_vol_window'], params['garch_p'], params['garch_q'])
        
        logger.debug(f"{asset_name}: Calculating Hurst exponent")
        hurst = calculate_hurst_exponent(prices, params['dfa_window'])
        
        # Integrate features
        logger.debug(f"{asset_name}: Determining final directions")
        directions_df = determine_final_direction(returns, ema_data, sma_data, predicted_volatility, hurst, params)
        
        # Calculate trend duration
        logger.debug(f"{asset_name}: Calculating trend duration")
        trend_duration = calculate_trend_duration(directions_df['direction'])
        directions_df['trend_duration'] = trend_duration
        
        # Generate pattern sequences
        logger.debug(f"{asset_name}: Generating pattern sequences")
        pattern_sequences = generate_pattern_sequences(
            directions_df, 
            params['pattern_min_length'], 
            params['pattern_max_length']
        )
        
        logger.info(f"{asset_name} processing completed: {len(directions_df)} directions, {len(pattern_sequences)} patterns")
        return directions_df, pattern_sequences
        
    except Exception as e:
        logger.error(f"{asset_name} processing failed: {e}")
        raise DirectionConverterError(f"{asset_name} processing failed: {e}")


# ===========================================
# VALIDATION AND TESTING
# ===========================================

def validate_direction_converter() -> None:
    """
    Validate direction converter functionality with test cases.
    """
    print("=== Direction Converter Validation ===")
    
    # Check available dependencies
    print("\n0. Checking dependencies:")
    print(f"  pandas: {'✓ Available' if PANDAS_AVAILABLE else '✗ Not available'}")
    print(f"  arch: {'✓ Available' if ARCH_AVAILABLE else '✗ Not available'}")
    print(f"  scipy/sklearn: {'✓ Available' if SCIPY_SKLEARN_AVAILABLE else '✗ Not available'}")
    
    if not PANDAS_AVAILABLE:
        print("\n⚠ pandas not available - performing basic validation only")
        
        # Test parameter validation without pandas
        print("\n1. Testing basic parameter validation:")
        try:
            # Valid parameters
            valid_params = {
                'ema_short_window': 12,
                'ema_long_window': 26,
                'volatility_k': 0.5
            }
            validated = validate_conversion_params(valid_params)
            print(f"  ✓ Parameter validation works: {len(validated)} params processed")
            
            # Test invalid parameters
            try:
                invalid_params = {
                    'ema_short_window': 30,  # Should be < ema_long_window (26)
                    'ema_long_window': 26
                }
                validate_conversion_params(invalid_params)
                print("  ✗ Invalid parameters not caught")
            except ValueError:
                print("  ✓ Invalid parameters correctly rejected")
                
        except Exception as e:
            print(f"  ✗ Parameter validation failed: {e}")
        
        print("\n✓ Basic validation completed without pandas")
        print("  To run full validation, install: pip install pandas numpy")
        return
    
    # Test parameter validation
    print("\n1. Testing parameter validation:")
    try:
        # Valid parameters
        valid_params = {
            'ema_short_window': 12,
            'ema_long_window': 26,
            'volatility_k': 0.5
        }
        validated = validate_conversion_params(valid_params)
        print(f"  ✓ Valid parameters accepted: {len(validated)} params")
        
        # Invalid parameters
        invalid_params = {
            'ema_short_window': 30,  # Should be < ema_long_window (26)
            'ema_long_window': 26
        }
        try:
            validate_conversion_params(invalid_params)
            print("  ✗ Invalid parameters not caught")
        except ValueError:
            print("  ✓ Invalid parameters correctly rejected")
            
    except Exception as e:
        print(f"  ✗ Parameter validation test failed: {e}")
    
    # Test trend calculations with mock data
    print("\n2. Testing trend calculations:")
    try:
        if PANDAS_AVAILABLE:
            # Create mock price data
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            prices = pd.Series(
                100 + np.cumsum(np.random.randn(100) * 0.02),  # Random walk with drift
                index=dates
            )
            
            # Test EMA calculation
            ema_result = calculate_ema_trend(prices, 12, 26)
            print(f"  ✓ EMA calculation: {len(ema_result)} components")
            
            # Test SMA calculation
            sma_result = calculate_sma_trend(prices, 20, 50)
            print(f"  ✓ SMA calculation: {len(sma_result)} components")
            
            # Check result structure
            required_ema_keys = ['ema_short', 'ema_long', 'trend_direction', 'trend_strength']
            if all(key in ema_result for key in required_ema_keys):
                print("  ✓ EMA result structure correct")
            else:
                print("  ✗ EMA result structure incorrect")
        else:
            print("  ⚠ Pandas not available - skipping trend calculation tests")
        
    except Exception as e:
        print(f"  ✗ Trend calculation test failed: {e}")
    
    # Test data insufficient error
    print("\n3. Testing error handling:")
    try:
        if PANDAS_AVAILABLE:
            short_prices = pd.Series([100, 101, 99], index=pd.date_range('2023-01-01', periods=3))
            try:
                calculate_ema_trend(short_prices, 12, 26)  # Should fail - insufficient data
                print("  ✗ Insufficient data error not raised")
            except DataInsufficientError:
                print("  ✓ Insufficient data error correctly raised")
            except Exception as e:
                print(f"  ✗ Unexpected error instead of DataInsufficientError: {e}")
        else:
            print("  ⚠ Pandas not available - skipping error handling tests")
    except Exception as e:
        print(f"  ✗ Error handling test setup failed: {e}")
    
    # Test GARCH functionality
    print("\n4. Testing GARCH volatility:")
    try:
        if PANDAS_AVAILABLE:
            # Create mock return data
            dates = pd.date_range('2023-01-01', periods=300, freq='D')
            returns = pd.Series(np.random.randn(300) * 0.02, index=dates)
            
            vol_result = calculate_garch_volatility(returns, 252, 1, 1)
            print(f"  ✓ GARCH volatility calculation: {len(vol_result)} values")
            
            if ARCH_AVAILABLE:
                print("  ✓ ARCH package available for full GARCH functionality")
            else:
                print("  ⚠ ARCH package not available - using historical volatility fallback")
        else:
            print("  ⚠ Pandas not available - skipping GARCH tests")
    except Exception as e:
        print(f"  ✗ GARCH test failed: {e}")
    
    # Test Hurst calculation
    print("\n5. Testing Hurst exponent:")
    try:
        if PANDAS_AVAILABLE:
            # Create mock price data with trend
            dates = pd.date_range('2023-01-01', periods=200, freq='D')
            trend_prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.01 + 0.001), index=dates)
            
            hurst_result = calculate_hurst_exponent(trend_prices, 100)
            print(f"  ✓ Hurst exponent calculation: {len(hurst_result)} values")
            
            if SCIPY_SKLEARN_AVAILABLE:
                print("  ✓ Scipy/sklearn available for full DFA functionality")
            else:
                print("  ⚠ Scipy/sklearn not available - using R/S method fallback")
        else:
            print("  ⚠ Pandas not available - skipping Hurst tests")
    except Exception as e:
        print(f"  ✗ Hurst test failed: {e}")
        
    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_direction_converter()