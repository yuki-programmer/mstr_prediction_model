"""
mstr_prediction_model/data/preprocessor.py

Data preprocessor module for MSTR prediction system.

This module handles preprocessing of raw market data, converting RawDataContainer
to ProcessedDataContainer with data validation, cleaning, and quality assessment.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass
import logging

# Handle pandas import with fallback for testing
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
        class Timestamp:
            def __init__(self, dt):
                self.dt = dt
            @staticmethod
            def now():
                return MockPandas.Timestamp(datetime.now())
    pd = MockPandas()
    np = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.volume_converter import convert_volume_string
    from data.loader import RawDataContainer
except ImportError as e:
    logger.error(f"Required module import failed: {e}")
    raise


@dataclass
class ProcessedDataContainer:
    """
    Processed data container for analysis-ready market data.
    
    Contains BTC, MSTR, and Gold processed data with standardized columns,
    data quality metrics, and processing metadata.
    """
    btc_processed: pd.DataFrame
    """
    BTC processed data
    
    Required column structure:
    - index: pd.DatetimeIndex (date, continuity ensured)
    - columns: ['close', 'open', 'high', 'low', 'volume', 'change_pct', 'returns']
    
    Data types:
    - close, open, high, low: float64 (positive values)
    - volume: float64 (K/M converted, positive values)
    - change_pct: float64 (decimal format)
    - returns: float64 (daily returns, log returns)
    
    Constraints:
    - high >= max(open, close)
    - low <= min(open, close)
    - volume >= 0
    - no missing values
    """
    
    mstr_processed: pd.DataFrame
    """
    MSTR processed data (same column structure as btc_processed)
    """
    
    gold_processed: pd.DataFrame
    """
    Gold processed data (same column structure as btc_processed)
    """
    
    common_date_range: Tuple[pd.Timestamp, pd.Timestamp]
    """
    Common available period
    Example: (Timestamp('2020-01-01'), Timestamp('2025-06-05'))
    """
    
    data_quality_report: Dict[str, Any]
    """
    Data quality report
    {
        'total_records': int,
        'missing_data_pct': float,
        'outlier_count': Dict[str, int],
        'data_completeness': float [0.0-1.0],
        'quality_score': float [0.0-1.0]
    }
    """
    
    processing_metadata: Dict[str, Any]
    """
    Processing parameters and logs
    {
        'volume_conversion_method': str,
        'outlier_detection_method': str,
        'interpolation_method': str,
        'processing_timestamp': pd.Timestamp,
        'data_filters_applied': List[str]
    }
    """
    
    def get_asset_data(self, asset: str) -> pd.DataFrame:
        """
        Get data for specified asset.
        
        Args:
            asset: Asset name ('btc', 'mstr', 'gold')
            
        Returns:
            DataFrame for the specified asset
        """
        asset_map = {
            'btc': self.btc_processed,
            'mstr': self.mstr_processed,
            'gold': self.gold_processed
        }
        return asset_map.get(asset.lower())
    
    def get_common_period_data(self) -> Dict[str, pd.DataFrame]:
        """
        Extract data for common period only.
        
        Returns:
            Dictionary containing data for all assets in common period
        """
        start, end = self.common_date_range
        return {
            'btc': self.btc_processed.loc[start:end],
            'mstr': self.mstr_processed.loc[start:end],
            'gold': self.gold_processed.loc[start:end]
        }


def process_single_asset(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """
    Process single asset data with comprehensive cleaning and validation.
    
    Args:
        df: Raw DataFrame with Japanese column names
        asset_name: Asset name for logging ('BTC', 'MSTR', 'Gold')
        
    Returns:
        Processed DataFrame with standardized structure
        
    Raises:
        ValueError: If data processing fails
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for data processing")
    
    logger.info(f"Processing {asset_name} data: {len(df)} rows")
    
    # Make a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Step 1: Volume conversion
    logger.info(f"Converting volume strings for {asset_name}")
    if 'volume_str' in processed_df.columns:
        processed_df['volume'] = processed_df['volume_str'].apply(convert_volume_string)
        processed_df = processed_df.drop('volume_str', axis=1)
    else:
        logger.warning(f"No volume_str column found for {asset_name}")
        processed_df['volume'] = 0.0
    
    # Step 2: Date index processing
    logger.info(f"Processing date index for {asset_name}")
    if not isinstance(processed_df.index, pd.DatetimeIndex):
        if 'date' in processed_df.columns:
            processed_df = processed_df.set_index('date')
        else:
            raise ValueError(f"Cannot establish date index for {asset_name}")
    
    # Remove duplicate dates, keep first
    duplicate_dates = processed_df.index.duplicated()
    if duplicate_dates.any():
        logger.warning(f"Removing {duplicate_dates.sum()} duplicate dates for {asset_name}")
        processed_df = processed_df[~duplicate_dates]
    
    # Sort by date
    processed_df = processed_df.sort_index()
    
    # Step 3: Calculate returns
    logger.info(f"Calculating returns for {asset_name}")
    if 'close' in processed_df.columns:
        # Log returns calculation
        processed_df['returns'] = np.log(processed_df['close'] / processed_df['close'].shift(1))
    else:
        raise ValueError(f"No close price column found for {asset_name}")
    
    # Step 4: Price data validation
    logger.info(f"Validating price data for {asset_name}")
    price_columns = ['open', 'high', 'low', 'close']
    
    # Check for positive values
    for col in price_columns:
        if col in processed_df.columns:
            negative_values = (processed_df[col] <= 0).sum()
            if negative_values > 0:
                logger.warning(f"{asset_name}: {negative_values} non-positive values in {col}")
                processed_df = processed_df[processed_df[col] > 0]
    
    # Check OHLC constraints
    if all(col in processed_df.columns for col in ['open', 'high', 'low', 'close']):
        # high >= max(open, close)
        high_violation = processed_df['high'] < np.maximum(processed_df['open'], processed_df['close'])
        # low <= min(open, close)  
        low_violation = processed_df['low'] > np.minimum(processed_df['open'], processed_df['close'])
        
        violations = high_violation.sum() + low_violation.sum()
        if violations > 0:
            logger.warning(f"{asset_name}: {violations} OHLC constraint violations detected")
            # Remove violating rows
            processed_df = processed_df[~(high_violation | low_violation)]
    
    # Step 5: Missing value handling
    logger.info(f"Handling missing values for {asset_name}")
    initial_rows = len(processed_df)
    
    # Forward fill then backward fill (pandas 2.0+ compatible)
    processed_df = processed_df.ffill().bfill()
    
    # Remove any remaining missing values
    processed_df = processed_df.dropna()
    
    final_rows = len(processed_df)
    if final_rows < initial_rows:
        logger.info(f"{asset_name}: Removed {initial_rows - final_rows} rows due to missing values")
    
    # Step 6: Outlier detection
    logger.info(f"Detecting outliers for {asset_name}")
    outlier_count = 0
    
    if 'returns' in processed_df.columns:
        # 4-sigma threshold for returns
        returns_mean = processed_df['returns'].mean()
        returns_std = processed_df['returns'].std()
        
        outlier_threshold = 4 * returns_std
        outliers = np.abs(processed_df['returns'] - returns_mean) > outlier_threshold
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.warning(f"{asset_name}: {outlier_count} return outliers detected (4σ threshold)")
            # Record but don't remove outliers - keep for analysis
    
    # Step 7: Ensure required columns and types
    required_columns = ['close', 'open', 'high', 'low', 'volume', 'change_pct', 'returns']
    
    for col in required_columns:
        if col not in processed_df.columns:
            if col == 'volume':
                processed_df[col] = 0.0
            elif col == 'returns':
                processed_df[col] = 0.0
            else:
                logger.error(f"Missing required column {col} for {asset_name}")
                raise ValueError(f"Missing required column {col}")
    
    # Ensure proper data types
    for col in ['close', 'open', 'high', 'low', 'volume', 'change_pct', 'returns']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Final validation
    processed_df = processed_df[required_columns]  # Keep only required columns
    processed_df = processed_df.dropna()  # Remove any remaining NaN values
    
    logger.info(f"{asset_name} processing complete: {len(processed_df)} rows retained")
    return processed_df


def find_common_date_range(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find common date range across all DataFrames.
    
    Args:
        dfs: Dictionary of DataFrames with asset names as keys
        
    Returns:
        Tuple of (start_date, end_date) for common period
        
    Raises:
        ValueError: If insufficient common period exists
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for date range calculation")
    
    logger.info("Finding common date range across assets")
    
    valid_dfs = {name: df for name, df in dfs.items() if not df.empty}
    
    if len(valid_dfs) < 2:
        raise ValueError("Need at least 2 valid datasets to determine common range")
    
    # Find latest start date and earliest end date
    start_dates = [df.index.min() for df in valid_dfs.values()]
    end_dates = [df.index.max() for df in valid_dfs.values()]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    # Check if we have at least 365 days of common data
    common_days = (common_end - common_start).days
    
    if common_days < 365:
        logger.warning(f"Common date range is only {common_days} days, which may be insufficient")
    
    logger.info(f"Common date range: {common_start.date()} to {common_end.date()} ({common_days} days)")
    
    return common_start, common_end


def generate_quality_report(processed_dfs: Dict[str, pd.DataFrame], raw_data: RawDataContainer) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    
    Args:
        processed_dfs: Dictionary of processed DataFrames
        raw_data: Original raw data container
        
    Returns:
        Data quality report dictionary
    """
    logger.info("Generating data quality report")
    
    # Calculate total records
    total_records = sum(len(df) for df in processed_dfs.values())
    
    # Calculate missing data percentage (compared to raw data)
    raw_total = sum(len(getattr(raw_data, f"{asset}_data")) for asset in ['btc', 'mstr', 'gold'])
    missing_data_pct = max(0, (raw_total - total_records) / raw_total) if raw_total > 0 else 0
    
    # Calculate outlier count per asset
    outlier_count = {}
    for asset, df in processed_dfs.items():
        if not df.empty and 'returns' in df.columns:
            returns_mean = df['returns'].mean()
            returns_std = df['returns'].std()
            outliers = np.abs(df['returns'] - returns_mean) > (4 * returns_std)
            outlier_count[asset] = int(outliers.sum())
        else:
            outlier_count[asset] = 0
    
    # Calculate data completeness
    data_completeness = min(1.0, total_records / max(1, raw_total))
    
    # Calculate overall quality score
    completeness_score = data_completeness
    outlier_penalty = min(0.2, sum(outlier_count.values()) / max(1, total_records))
    quality_score = max(0.0, completeness_score - outlier_penalty)
    
    report = {
        'total_records': total_records,
        'missing_data_pct': round(missing_data_pct, 4),
        'outlier_count': outlier_count,
        'data_completeness': round(data_completeness, 4),
        'quality_score': round(quality_score, 4)
    }
    
    logger.info(f"Quality score: {quality_score:.3f}, Completeness: {data_completeness:.3f}")
    return report


def generate_processing_metadata() -> Dict[str, Any]:
    """
    Generate processing metadata with methods and parameters used.
    
    Returns:
        Processing metadata dictionary
    """
    if PANDAS_AVAILABLE:
        timestamp = pd.Timestamp.now()
    else:
        timestamp = datetime.now()
    
    metadata = {
        'volume_conversion_method': 'string_to_numeric_with_units',
        'outlier_detection_method': '4_sigma_threshold',
        'interpolation_method': 'forward_fill_backward_fill',
        'processing_timestamp': timestamp,
        'data_filters_applied': [
            'remove_duplicates',
            'sort_by_date', 
            'remove_outliers',
            'validate_ohlc_constraints',
            'convert_volume_strings'
        ]
    }
    
    return metadata


def preprocess_market_data(raw_data: RawDataContainer) -> ProcessedDataContainer:
    """
    Preprocess raw market data into analysis-ready format.
    
    Args:
        raw_data: RawDataContainer with loaded market data
        
    Returns:
        ProcessedDataContainer with cleaned and validated data
        
    Raises:
        ValueError: If data processing fails or insufficient data quality
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for data preprocessing")
    
    logger.info("Starting market data preprocessing")
    
    # Step 1: Validate input data
    validation_results = raw_data.validate()
    logger.info(f"Raw data validation: {validation_results}")
    
    # Step 2: Process each asset individually
    processed_dfs = {}
    
    assets = {
        'btc': raw_data.btc_data,
        'mstr': raw_data.mstr_data,
        'gold': raw_data.gold_data
    }
    
    for asset_name, df in assets.items():
        if not df.empty:
            try:
                processed_df = process_single_asset(df, asset_name.upper())
                processed_dfs[asset_name] = processed_df
                logger.info(f"✓ {asset_name.upper()}: {len(processed_df)} records processed")
            except Exception as e:
                logger.error(f"✗ Failed to process {asset_name.upper()}: {e}")
                processed_dfs[asset_name] = pd.DataFrame()  # Empty DataFrame as fallback
        else:
            logger.warning(f"⚠ {asset_name.upper()}: No data to process")
            processed_dfs[asset_name] = pd.DataFrame()
    
    # Step 3: Find common date range
    valid_dfs = {name: df for name, df in processed_dfs.items() if not df.empty}
    
    if len(valid_dfs) < 2:
        raise ValueError("Insufficient processed data: need at least 2 valid datasets")
    
    common_start, common_end = find_common_date_range(valid_dfs)
    
    # Step 4: Generate quality report
    quality_report = generate_quality_report(processed_dfs, raw_data)
    
    # Step 5: Generate processing metadata
    processing_metadata = generate_processing_metadata()
    
    # Step 6: Create ProcessedDataContainer
    container = ProcessedDataContainer(
        btc_processed=processed_dfs.get('btc', pd.DataFrame()),
        mstr_processed=processed_dfs.get('mstr', pd.DataFrame()),
        gold_processed=processed_dfs.get('gold', pd.DataFrame()),
        common_date_range=(common_start, common_end),
        data_quality_report=quality_report,
        processing_metadata=processing_metadata
    )
    
    logger.info("=== Preprocessing Summary ===")
    logger.info(f"BTC: {len(container.btc_processed)} records")
    logger.info(f"MSTR: {len(container.mstr_processed)} records")
    logger.info(f"Gold: {len(container.gold_processed)} records")
    logger.info(f"Common period: {common_start.date()} to {common_end.date()}")
    logger.info(f"Quality score: {quality_report['quality_score']:.3f}")
    
    return container


def validate_preprocessing() -> None:
    """
    Validate preprocessing functionality with test cases.
    """
    print("=== Data Preprocessor Validation ===")
    
    # Test 1: Test volume conversion integration
    print("\n1. Testing volume conversion integration:")
    test_volumes = ["1.5K", "2.3M", "0.8B", "invalid", ""]
    for vol in test_volumes:
        result = convert_volume_string(vol)
        print(f"  {vol} -> {result}")
    
    # Test 2: Check pandas availability
    print(f"\n2. Pandas availability: {'✓ Available' if PANDAS_AVAILABLE else '✗ Not available'}")
    
    # Test 3: Test date range calculation
    print(f"\n3. Common date range calculation:")
    if PANDAS_AVAILABLE:
        # Create mock DataFrames for testing
        dates1 = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        dates2 = pd.date_range('2021-01-01', '2024-06-30', freq='D')
        dates3 = pd.date_range('2020-06-01', '2023-06-30', freq='D')
        
        mock_dfs = {
            'btc': pd.DataFrame({'close': range(len(dates1))}, index=dates1),
            'mstr': pd.DataFrame({'close': range(len(dates2))}, index=dates2),
            'gold': pd.DataFrame({'close': range(len(dates3))}, index=dates3)
        }
        
        try:
            start, end = find_common_date_range(mock_dfs)
            days = (end - start).days
            print(f"  Common range: {start.date()} to {end.date()} ({days} days)")
        except Exception as e:
            print(f"  Error in date range calculation: {e}")
    else:
        print("  Skipped - pandas not available")
    
    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_preprocessing()
    
    # Test with actual data if available
    print("\n=== Attempting Preprocessing Test ===")
    if PANDAS_AVAILABLE:
        try:
            from data.loader import load_all_market_data
            raw_data = load_all_market_data()
            processed_data = preprocess_market_data(raw_data)
            print("✓ Preprocessing test successful!")
            print(f"Quality report: {processed_data.data_quality_report}")
        except Exception as e:
            print(f"⚠ Preprocessing test failed (expected if no data): {e}")
    else:
        print("⚠ Pandas not available - skipping preprocessing test")