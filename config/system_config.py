"""
System configuration module for MSTR prediction system.

This module provides centralized configuration management for all system components,
including prediction parameters, data quality thresholds, analysis settings, and
runtime configurations with environment variable override support.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentMode(Enum):
    """Environment mode definitions."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class RiskProfile(Enum):
    """Investment risk profile definitions."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class SystemConfig:
    """
    System-wide configuration management with environment override support.
    
    Provides centralized access to all configuration parameters across
    the MSTR prediction system modules.
    """
    
    # Environment and Runtime Settings
    environment_mode: EnvironmentMode = field(default=EnvironmentMode.DEVELOPMENT)
    debug_mode: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    enable_profiling: bool = field(default=False)
    
    def __post_init__(self):
        """Initialize configuration with environment variable overrides."""
        self._apply_environment_overrides()
        self._validate_configuration()
        
        if self.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"SystemConfig initialized in {self.environment_mode.value} mode")

    # ===========================================
    # 1. PREDICTION PERIODS & TIME CONFIGURATION
    # ===========================================
    
    @property
    def PREDICTION_PERIODS(self) -> Dict[str, str]:
        """Standard prediction period definitions."""
        return {
            "ULTRA_SHORT": "7d",
            "SHORT": "14d",
            "MEDIUM_SHORT": "30d", 
            "MEDIUM": "90d",
            "MEDIUM_LONG": "180d"
        }
    
    @property
    def PERIOD_DAYS(self) -> Dict[str, int]:
        """Mapping of period strings to day counts."""
        return {
            "7d": 7,
            "14d": 14,
            "30d": 30,
            "90d": 90,
            "180d": 180
        }
    
    @property
    def PERIOD_PRIORITY(self) -> List[str]:
        """Priority order for prediction periods (short to long)."""
        return ["7d", "14d", "30d", "90d", "180d"]

    # ===========================================
    # 2. DATA QUALITY & VALIDATION THRESHOLDS
    # ===========================================
    
    @property
    def DATA_QUALITY_THRESHOLDS(self) -> Dict[str, float]:
        """Data quality validation thresholds."""
        return {
            'min_data_completeness': 0.95,      # Minimum 95% data completeness
            'max_missing_ratio': 0.05,          # Maximum 5% missing data
            'min_quality_score': 0.8,           # Minimum 80% quality score
            'max_outlier_ratio': 0.02,          # Maximum 2% outliers
            'min_confidence_threshold': 0.6,    # Minimum 60% confidence
        }
    
    @property
    def NORMALIZATION_RANGES(self) -> Dict[str, Tuple[float, float]]:
        """Normalization ranges for different data types."""
        return {
            'confidence_score': (0.0, 1.0),
            'probability': (0.0, 1.0),
            'correlation': (-1.0, 1.0),
            'returns': (-1.0, 1.0),
        }

    # ===========================================
    # 3. PATTERN ANALYSIS PARAMETERS
    # ===========================================
    
    @property
    def CONVERSION_PARAMS(self) -> Dict[str, Union[float, int]]:
        """Direction conversion parameters for pattern analysis."""
        return {
            'strength_threshold': 0.02,         # 2% minimum movement threshold
            'volatility_window': 20,            # 20-day volatility calculation window
            'trend_min_duration': 2,            # Minimum 2 days for trend
            'pattern_min_length': 3,            # Minimum pattern length
            'pattern_max_length': 10,           # Maximum pattern length
            'smoothing_window': 3,              # 3-day smoothing window
        }
    
    @property
    def MATCHING_PARAMS(self) -> Dict[str, Union[float, str, bool, int]]:
        """Pattern matching algorithm parameters."""
        return {
            'similarity_threshold': 0.7,        # 70% minimum similarity
            'matching_algorithm': 'cosine',     # Cosine similarity algorithm
            'normalization_method': 'minmax',   # Min-max normalization
            'window_size': 30,                  # 30-day matching window
            'allow_inverse_patterns': True,     # Allow inverse pattern matching
        }
    
    @property
    def LAG_PARAMS(self) -> Dict[str, Union[int, float]]:
        """Lag analysis parameters for BTC-MSTR correlation."""
        return {
            'max_lag_days': 30,                 # Maximum 30-day lag analysis
            'min_correlation': 0.1,             # Minimum 10% correlation
            'confidence_level': 0.95,           # 95% confidence level
            'rolling_window': 252,              # 252-day rolling window (1 year)
            'stability_threshold': 0.8,         # 80% stability requirement
        }
    
    @property
    def ANALYSIS_PARAMS(self) -> Dict[str, Union[float, str, bool]]:
        """Multi-period analysis parameters."""
        return {
            'consistency_threshold': 0.7,       # 70% consistency requirement
            'contradiction_sensitivity': 0.8,   # 80% contradiction detection
            'aggregation_method': 'weighted',   # Weighted aggregation method
            'confidence_weighting': True,       # Enable confidence weighting
            'temporal_decay': 0.95,             # 95% temporal decay factor
        }

    # ===========================================
    # 4. BTC PREDICTION & PHYSICAL MODEL PARAMS
    # ===========================================
    
    @property
    def PHYSICAL_MODEL_PARAMS(self) -> Dict[str, Union[str, int, float]]:
        """Physical model parameters for BTC prediction."""
        return {
            'model_type': 'hybrid',             # Hybrid physical model
            'fitting_period': 365,              # 365-day fitting period
            'validation_split': 0.2,            # 20% validation split
            'max_iterations': 1000,             # Maximum 1000 iterations
            'convergence_threshold': 1e-6,      # Convergence threshold
            'regularization_strength': 0.01,    # L2 regularization strength
        }
    
    @property
    def BTC_PREDICTION_PARAMS(self) -> Dict[str, Union[List[str], List[float], int, bool]]:
        """BTC prediction engine parameters."""
        return {
            'forecast_periods': ['7d', '14d', '30d', '90d', '180d'],
            'confidence_levels': [0.8, 0.95],   # 80% and 95% confidence levels
            'simulation_count': 10000,          # 10,000 Monte Carlo simulations
            'scenario_count': 3,                # 3 scenario analysis (bull/base/bear)
            'volatility_adjustment': True,      # Enable volatility adjustment
            'regime_awareness': True,           # Enable regime awareness
        }

    # ===========================================
    # 5. PERIOD-SPECIFIC CONFIGURATIONS
    # ===========================================
    
    @property
    def PERIOD_DEFAULT_CONFIGS(self) -> Dict[str, Dict[str, Union[float, bool]]]:
        """Period-specific default configurations."""
        return {
            '7d': {
                'btc_influence_factor': 0.8,
                'pattern_influence_factor': 0.6,
                'volatility_adjustment': True,
                'regime_awareness': True,
            },
            '14d': {
                'btc_influence_factor': 0.75,
                'pattern_influence_factor': 0.7,
                'volatility_adjustment': True,
                'regime_awareness': True,
            },
            '30d': {
                'btc_influence_factor': 0.7,
                'pattern_influence_factor': 0.75,
                'volatility_adjustment': True,
                'regime_awareness': True,
            },
            '90d': {
                'btc_influence_factor': 0.6,
                'pattern_influence_factor': 0.8,
                'volatility_adjustment': False,
                'regime_awareness': True,
            },
            '180d': {
                'btc_influence_factor': 0.5,
                'pattern_influence_factor': 0.85,
                'volatility_adjustment': False,
                'regime_awareness': False,
            }
        }

    # ===========================================
    # 6. INVESTMENT DECISION SETTINGS
    # ===========================================
    
    @property
    def DEFAULT_INVESTMENT_SETTINGS(self) -> Dict[str, Dict[str, float]]:
        """Risk profile-based investment settings."""
        return {
            'conservative': {
                'position_size_limit': 0.05,        # 5% max position size
                'stop_loss_threshold': 0.10,        # 10% stop loss
                'take_profit_threshold': 1.25,      # 25% take profit
                'max_drawdown_tolerance': 0.15,     # 15% max drawdown
            },
            'moderate': {
                'position_size_limit': 0.10,        # 10% max position size
                'stop_loss_threshold': 0.15,        # 15% stop loss
                'take_profit_threshold': 1.50,      # 50% take profit
                'max_drawdown_tolerance': 0.25,     # 25% max drawdown
            },
            'aggressive': {
                'position_size_limit': 0.20,        # 20% max position size
                'stop_loss_threshold': 0.20,        # 20% stop loss
                'take_profit_threshold': 2.00,      # 100% take profit
                'max_drawdown_tolerance': 0.40,     # 40% max drawdown
            }
        }

    # ===========================================
    # 7. DATA SCHEMA & COLUMN DEFINITIONS
    # ===========================================
    
    @property
    def OHLCV_COLUMNS(self) -> List[str]:
        """Standard OHLCV column names."""
        return ['open', 'high', 'low', 'close', 'volume']
    
    @property
    def PRICE_COLUMNS(self) -> List[str]:
        """Price-only column names."""
        return ['open', 'high', 'low', 'close']
    
    @property
    def DERIVED_COLUMNS(self) -> List[str]:
        """Derived/calculated column names."""
        return ['returns', 'change_pct']
    
    @property
    def DATE_INDEX_NAME(self) -> str:
        """Standard date index name."""
        return 'date'
    
    @property
    def ASSET_NAMES(self) -> Dict[str, str]:
        """Asset name standardization mapping."""
        return {
            'bitcoin': 'BTC',
            'btc': 'BTC',
            'microstrategy': 'MSTR',
            'mstr': 'MSTR',
            'gold': 'Gold'
        }

    # ===========================================
    # 8. FACTOR ANALYSIS COLUMN DEFINITIONS
    # ===========================================
    
    @property
    def TECHNICAL_COLUMNS(self) -> List[str]:
        """Technical analysis factor columns."""
        return [
            'momentum_score', 'trend_strength', 'volatility_score', 
            'support_resistance_score', 'volume_score'
        ]
    
    @property
    def FUNDAMENTAL_COLUMNS(self) -> List[str]:
        """Fundamental analysis factor columns."""
        return [
            'btc_correlation_score', 'company_health_score', 'market_sentiment_score',
            'regulatory_score', 'valuation_score'
        ]
    
    @property
    def PATTERN_COLUMNS(self) -> List[str]:
        """Pattern analysis factor columns."""
        return [
            'pattern_reliability_score', 'lag_accuracy_score', 'historical_success_score',
            'pattern_consistency_score', 'regime_adaptation_score'
        ]
    
    @property
    def CYCLE_COLUMNS(self) -> List[str]:
        """Cycle analysis factor columns."""
        return [
            'cycle_position_score', 'seasonal_score', 'regime_score',
            'halving_effect_score', 'macro_cycle_score'
        ]

    # ===========================================
    # 9. MINIMUM REQUIREMENTS & VALIDATION
    # ===========================================
    
    @property
    def MINIMUM_REQUIREMENTS(self) -> Dict[str, Union[float, int]]:
        """Minimum requirements for system operation."""
        return {
            'overall_confidence': 0.6,          # 60% minimum overall confidence
            'overall_data_quality': 0.8,        # 80% minimum data quality
            'sample_size': 30,                  # Minimum 30 data points
            'pattern_coverage': 0.8,            # 80% pattern coverage
            'high_quality_matches': 10,         # Minimum 10 high-quality matches
        }

    # ===========================================
    # 10. FILE & DATA SOURCE CONFIGURATION
    # ===========================================
    
    @property
    def DATA_SOURCE_CONFIG(self) -> Dict[str, Dict[str, str]]:
        """Data source file configuration."""
        return {
            'btc': {
                'file': 'BTC_USD_daily.xlsx',
                'sheet': 'BTC_USD Bitfinex 過去データ'
            },
            'mstr': {
                'file': 'MSTR_daily.xlsx',
                'sheet': 'MSTR 過去データ (2)'
            },
            'gold': {
                'file': 'gold_daily.xlsx',
                'sheet': '金先物 過去データ'
            }
        }
    
    @property
    def LOGGING_CONFIG(self) -> Dict[str, Any]:
        """Logging configuration settings."""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                }
            },
            'handlers': {
                'default': {
                    'level': log_level,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                },
                'file': {
                    'level': log_level,
                    'formatter': 'detailed',
                    'class': 'logging.FileHandler',
                    'filename': 'mstr_prediction.log',
                    'mode': 'a',
                }
            },
            'loggers': {
                '': {
                    'handlers': ['default'],
                    'level': log_level,
                    'propagate': False
                }
            }
        }

    # ===========================================
    # ENVIRONMENT & VALIDATION METHODS
    # ===========================================
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Environment mode override
        env_mode = os.getenv('MSTR_ENVIRONMENT', '').lower()
        if env_mode in ['development', 'dev']:
            self.environment_mode = EnvironmentMode.DEVELOPMENT
        elif env_mode in ['testing', 'test']:
            self.environment_mode = EnvironmentMode.TESTING
        elif env_mode in ['production', 'prod']:
            self.environment_mode = EnvironmentMode.PRODUCTION
        
        # Debug mode override
        debug_env = os.getenv('MSTR_DEBUG', '').lower()
        if debug_env in ['true', '1', 'yes']:
            self.debug_mode = True
        elif debug_env in ['false', '0', 'no']:
            self.debug_mode = False
        
        # Verbose logging override
        verbose_env = os.getenv('MSTR_VERBOSE', '').lower()
        if verbose_env in ['true', '1', 'yes']:
            self.verbose_logging = True
        
        # Profiling override
        profile_env = os.getenv('MSTR_PROFILING', '').lower()
        if profile_env in ['true', '1', 'yes']:
            self.enable_profiling = True
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Validate prediction periods
        if not all(period in self.PERIOD_DAYS for period in self.PERIOD_PRIORITY):
            errors.append("Inconsistency between PERIOD_PRIORITY and PERIOD_DAYS")
        
        # Validate thresholds are in valid ranges
        for key, value in self.DATA_QUALITY_THRESHOLDS.items():
            if not 0.0 <= value <= 1.0:
                errors.append(f"DATA_QUALITY_THRESHOLDS[{key}] must be between 0.0 and 1.0")
        
        # Validate normalization ranges
        for key, (min_val, max_val) in self.NORMALIZATION_RANGES.items():
            if min_val >= max_val:
                errors.append(f"NORMALIZATION_RANGES[{key}] min >= max")
        
        # Validate investment settings
        for profile, settings in self.DEFAULT_INVESTMENT_SETTINGS.items():
            for setting, value in settings.items():
                if value < 0:
                    errors.append(f"Investment setting {profile}.{setting} cannot be negative")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get_period_config(self, period: str) -> Dict[str, Union[float, bool]]:
        """
        Get configuration for specific prediction period.
        
        Args:
            period: Prediction period string (e.g., '7d', '30d')
            
        Returns:
            Period-specific configuration dictionary
        """
        if period not in self.PERIOD_DEFAULT_CONFIGS:
            logger.warning(f"Unknown period {period}, using default 30d config")
            period = '30d'
        
        return self.PERIOD_DEFAULT_CONFIGS[period].copy()
    
    def get_investment_settings(self, risk_profile: Union[str, RiskProfile]) -> Dict[str, float]:
        """
        Get investment settings for specific risk profile.
        
        Args:
            risk_profile: Risk profile (conservative/moderate/aggressive)
            
        Returns:
            Investment settings dictionary
        """
        if isinstance(risk_profile, RiskProfile):
            profile = risk_profile.value
        else:
            profile = str(risk_profile).lower()
        
        if profile not in self.DEFAULT_INVESTMENT_SETTINGS:
            logger.warning(f"Unknown risk profile {profile}, using moderate")
            profile = 'moderate'
        
        return self.DEFAULT_INVESTMENT_SETTINGS[profile].copy()
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode."""
        return self.environment_mode == EnvironmentMode.PRODUCTION
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug_mode
    
    def get_data_directory(self) -> Path:
        """Get data directory path based on environment."""
        if self.environment_mode == EnvironmentMode.PRODUCTION:
            return Path("/data/mstr_prediction")
        else:
            return Path("data")


# Global configuration instance
CONFIG = SystemConfig()


def get_config() -> SystemConfig:
    """
    Get global configuration instance.
    
    Returns:
        SystemConfig instance
    """
    return CONFIG


def validate_system_config() -> None:
    """
    Validate system configuration with comprehensive tests.
    """
    print("=== System Configuration Validation ===")
    
    config = get_config()
    
    # Test 1: Basic configuration access
    print(f"\n1. Environment: {config.environment_mode.value}")
    print(f"   Debug mode: {config.debug_mode}")
    print(f"   Prediction periods: {len(config.PERIOD_PRIORITY)}")
    
    # Test 2: Validation checks
    print("\n2. Configuration validation:")
    try:
        config._validate_configuration()
        print("   ✓ All validation checks passed")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
    
    # Test 3: Period configuration access
    print("\n3. Period configuration test:")
    for period in ['7d', '30d', '180d']:
        period_config = config.get_period_config(period)
        print(f"   {period}: BTC influence = {period_config['btc_influence_factor']}")
    
    # Test 4: Investment settings
    print("\n4. Investment settings test:")
    for profile in ['conservative', 'moderate', 'aggressive']:
        settings = config.get_investment_settings(profile)
        print(f"   {profile}: max position = {settings['position_size_limit']*100}%")
    
    # Test 5: Environment overrides
    print(f"\n5. Environment overrides:")
    print(f"   MSTR_ENVIRONMENT: {os.getenv('MSTR_ENVIRONMENT', 'not set')}")
    print(f"   MSTR_DEBUG: {os.getenv('MSTR_DEBUG', 'not set')}")
    
    # Test 6: Data structure validation
    print(f"\n6. Data structure validation:")
    print(f"   OHLCV columns: {len(config.OHLCV_COLUMNS)}")
    print(f"   Technical factors: {len(config.TECHNICAL_COLUMNS)}")
    print(f"   Data sources: {len(config.DATA_SOURCE_CONFIG)}")
    
    print(f"\n=== Validation Complete ===")


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_system_config()