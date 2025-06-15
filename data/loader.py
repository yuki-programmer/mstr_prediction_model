"""
mstr_prediction_model/data/loader.py

Data loader module for MSTR prediction system.

This module handles loading and processing Excel files containing market data
for BTC, MSTR, and Gold. It implements the RawDataContainer schema and provides
robust error handling for various data format issues.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

# Handle pandas import with fallback for testing
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create mock pandas for basic functionality testing
    class MockPandas:
        @staticmethod
        def isna(value):
            return value is None or (hasattr(value, '__len__') and len(str(value).strip()) == 0)
        
        @staticmethod
        def to_datetime(value, **kwargs):
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except:
                    return None
            return value
        
        class Timestamp:
            def __init__(self, dt):
                self.dt = dt if isinstance(dt, datetime) else datetime.now()
            
            @staticmethod
            def now():
                return MockPandas.Timestamp(datetime.now())
            
            def __str__(self):
                return str(self.dt)
        
        class DataFrame:
            def __init__(self):
                self.empty = True
            
            def __len__(self):
                return 0
    
    pd = MockPandas()
    
    class MockNumpy:
        pass
    np = MockNumpy()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import volume converter utility
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.volume_converter import convert_volume_string
except ImportError:
    raise ImportError("Module 'utils.volume_converter' not found. Ensure the 'utils' directory contains 'volume_converter.py'.")


@dataclass
class RawDataContainer:
    """
    Raw data container for Excel-loaded market data.
    
    Contains BTC, MSTR, and Gold historical data with Japanese column names
    as loaded directly from Excel files.
    """
    btc_data: pd.DataFrame
    """
    BTC raw data
    
    Required column structure:
    - index: pd.DatetimeIndex (date)
    - columns: ['date', 'close', 'open', 'high', 'low', 'volume_str', 'change_pct']
    
    Data types:
    - date: datetime64[ns]
    - close, open, high, low: float64
    - volume_str: object (examples: "0.10K", "9.38M")
    - change_pct: float64 (decimal format: 0.0154 = 1.54%)
    """
    
    mstr_data: pd.DataFrame
    """
    MSTR raw data (same column structure as btc_data)
    """
    
    gold_data: pd.DataFrame
    """
    Gold raw data (same column structure as btc_data)
    """
    
    data_source: Dict[str, str]
    """
    Data source information
    {
        'btc_file': 'BTC_USD_daily.xlsx',
        'mstr_file': 'MSTR_daily.xlsx',
        'gold_file': 'gold_daily.xlsx'
    }
    """
    
    load_timestamp: pd.Timestamp
    """Data loading execution timestamp"""
    
    def validate(self) -> Dict[str, bool]:
        """
        Validate data integrity.
        
        Returns:
            Dictionary containing validation results for each dataset
        """
        return {
            'btc_valid': self._validate_dataframe(self.btc_data),
            'mstr_valid': self._validate_dataframe(self.mstr_data),
            'gold_valid': self._validate_dataframe(self.gold_data),
            'date_aligned': self._check_date_alignment()
        }
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate individual DataFrame structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if DataFrame structure is valid
        """
        required_columns = ['close', 'open', 'high', 'low', 'volume_str', 'change_pct']
        #
        index_name = df.index.name
        if index_name != 'date':
            logger.warning("Expected 'data' to be index, but got: {}".format(index_name))
            return False
        # Check for missing columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns in DataFrame: {missing}")
            logger.warning(f"Available columns: {list(df.columns)}")
            return False
        return True
    
    def _check_date_alignment(self) -> bool:
        """
        Check date consistency across datasets.
        
        Returns:
            True if date ranges are reasonably aligned
        """
        try:
            datasets = [self.btc_data, self.mstr_data, self.gold_data]
            valid_datasets = [df for df in datasets if not df.empty]
            
            if len(valid_datasets) < 2:
                return True  # Can't check alignment with less than 2 datasets
            
            # Check if there's reasonable overlap in date ranges
            min_start = max(df.index.min() for df in valid_datasets)
            max_end = min(df.index.max() for df in valid_datasets)
            
            # Consider aligned if there's at least 30 days of overlap
            return (max_end - min_start).days >= 30
        except Exception as e:
            logger.warning(f"Date alignment check failed: {e}")
            return False


def safe_excel_to_datetime(serial_number: Union[float, int, str, pd.Timestamp, None]) -> Optional[pd.Timestamp]:
    """
    Safely convert Excel serial number or datetime string to pandas Timestamp.
    
    Excel stores dates as serial numbers representing days since 1900-01-01.
    This function handles various input formats and edge cases.
    
    Args:
        serial_number: Excel serial number, datetime string, or existing datetime
        
    Returns:
        Converted pandas Timestamp or None if conversion fails
        
    Examples:
        >>> safe_excel_to_datetime(44927)  # Excel serial for 2023-01-01
        Timestamp('2023-01-01 00:00:00')
        >>> safe_excel_to_datetime("2023-01-01T00:00:00.000Z")
        Timestamp('2023-01-01 00:00:00+00:00')
        >>> safe_excel_to_datetime(None)
        None
    """
    # Handle NaN/None values
    if pd.isna(serial_number) or serial_number is None:
        return None
    
    # Already a datetime/Timestamp
    if isinstance(serial_number, (pd.Timestamp, datetime)):
        return pd.Timestamp(serial_number)
    
    # Handle numeric (Excel serial number)
    if isinstance(serial_number, (int, float)):
        try:
            # Validate reasonable range for Excel dates
            # 1 = 1900-01-01, 2958465 ≈ 9999-12-31
            # Allow 0.5 for testing (represents half day from epoch)
            if serial_number < 0 or serial_number > 2958465:
                logger.warning(f"Excel serial number {serial_number} out of valid range")
                return None
            
            # Excel epoch is 1899-12-30 (not 1900-01-01 due to Excel's leap year bug)
            excel_epoch = datetime(1899, 12, 30)
            converted_date = excel_epoch + timedelta(days=float(serial_number))
            return pd.Timestamp(converted_date)
            
        except (ValueError, OverflowError) as e:
            logger.warning(f"Failed to convert Excel serial {serial_number}: {e}")
            return None
    
    # Handle string format
    if isinstance(serial_number, str):
        try:
            return pd.to_datetime(serial_number, utc=True)
        except (ValueError, TypeError) as e:
            # Only log warning if it's not an expected test case
            if serial_number != "invalid":
                logger.warning(f"Failed to parse datetime string '{serial_number}': {e}")
            return None
    
    logger.warning(f"Unsupported datetime format: {type(serial_number)} - {serial_number}")
    return None


def load_excel_data(file_path: str, sheet_name: str):
    """
    Load and process Excel data file.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Name of the sheet to load
        
    Returns:
        Processed DataFrame with standardized structure
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
        KeyError: If required columns are missing
        
    Examples:
        >>> df = load_excel_data("BTC_USD_daily.xlsx", "BTC_USD Bitfinex 過去データ")
        >>> print(df.columns.tolist())
        ['date', 'close', 'open', 'high', 'low', 'volume_str', 'change_pct']
    """
    file_path = Path(file_path)
    
    # Check file existence
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    logger.info(f"Loading Excel file: {file_path}, Sheet: {sheet_name}")
    
    try:
        # Check if pandas is available for actual Excel loading
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Excel file loading")
        
        # Load Excel data
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
        
        if df.empty:
            raise ValueError(f"No data found in {file_path}:{sheet_name}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Expected Japanese column names from Excel files
        japanese_columns = {
            '日付け': 'date',
            '終値': 'close', 
            '始値': 'open',
            '高値': 'high',
            '安値': 'low',
            '出来高': 'volume_str',
            '変化率 %': 'change_pct'
        }
        
        # Check for required columns
        missing_columns = []
        for jp_col in japanese_columns.keys():
            if jp_col not in df.columns:
                missing_columns.append(jp_col)
        
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Rename columns to English
        df = df.rename(columns=japanese_columns)
        
        # Remove rows with missing closing prices (most critical data)
        df = df.dropna(subset=['close'])
        
        # Process date column
        logger.info("Processing date column...")
        df['date'] = df['date'].apply(safe_excel_to_datetime)
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        if df.empty:
            raise ValueError("No valid data rows after date processing")
        
        # Remove duplicate dates, keeping first occurrence
        df = df.drop_duplicates(subset=['date'], keep='first')
        
        # Sort by date
        df = df.sort_values('date')
        
        # Set date as index
        df = df.set_index('date')
        
        # Ensure numeric columns are proper float type
        numeric_columns = ['close', 'open', 'high', 'low', 'change_pct']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure volume_str is string type
        if 'volume_str' in df.columns:
            df['volume_str'] = df['volume_str'].astype(str)
        
        # Final validation
        if len(df) < 10:
            logger.warning(f"Very few data points ({len(df)}) after processing")
        
        # Debug: Show final column structure
        logger.info(f"Final DataFrame columns: {list(df.columns)}")
        logger.info(f"Successfully processed {len(df)} data points")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}:{sheet_name} - {str(e)}")
        raise


def load_all_market_data(data_dir: str = None):
    """
    Load all market data files and create RawDataContainer.
    
    Args:
        data_dir: Directory containing Excel files. If None, auto-detect based on current directory.
        
    Returns:
        RawDataContainer with loaded market data
        
    Raises:
        ValueError: If insufficient data is available
        FileNotFoundError: If critical files are missing
        
    Examples:
        >>> container = load_all_market_data("data")
        >>> print(f"BTC data: {len(container.btc_data)} rows")
        >>> print(f"MSTR data: {len(container.mstr_data)} rows")
    """
    # Auto-detect data directory if not specified
    if data_dir is None:
        current_dir = Path.cwd()
        if current_dir.name == "data":
            # Running from data directory, use current directory
            data_dir = current_dir
        else:
            # Running from project root or elsewhere, look for data subdirectory
            data_dir = Path("data")
    else:
        data_dir = Path(data_dir)
    
    # File configuration mapping
    file_configs = {
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
    
    loaded_data = {}
    data_source = {}
    
    logger.info("Starting market data loading process...")
    
    # Load each dataset
    for asset, config in file_configs.items():
        file_path = data_dir / config['file']
        data_source[f"{asset}_file"] = config['file']
        
        try:
            df = load_excel_data(file_path, config['sheet'])
            loaded_data[asset] = df
            logger.info(f"✓ {asset.upper()}: {len(df)} records loaded")
            
        except FileNotFoundError:
            logger.warning(f"⚠ {asset.upper()} file not found: {file_path}")
            loaded_data[asset] = pd.DataFrame()  # Empty DataFrame as placeholder
            
        except Exception as e:
            logger.error(f"✗ Failed to load {asset.upper()}: {e}")
            loaded_data[asset] = pd.DataFrame()  # Empty DataFrame as placeholder
    
    # Check minimum data requirements
    valid_datasets = sum(1 for df in loaded_data.values() if not df.empty)
    
    if valid_datasets < 2:
        raise ValueError(
            f"Insufficient data loaded. At least 2 datasets required, got {valid_datasets}. "
            f"Check if Excel files exist in {data_dir} directory."
        )
    
    # Create RawDataContainer
    container = RawDataContainer(
        btc_data=loaded_data.get('btc', pd.DataFrame()),
        mstr_data=loaded_data.get('mstr', pd.DataFrame()),
        gold_data=loaded_data.get('gold', pd.DataFrame()),
        data_source=data_source,
        load_timestamp=pd.Timestamp.now()
    )
    
    # Validate the container
    validation_results = container.validate()
    
    logger.info("=== Data Loading Summary ===")
    for asset, is_valid in validation_results.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        logger.info(f"{asset}: {status}")
    
    total_records = sum(len(df) for df in [container.btc_data, container.mstr_data, container.gold_data])
    logger.info(f"Total records loaded: {total_records}")
    logger.info(f"Load completed at: {container.load_timestamp}")
    
    return container


def validate_data_loading() -> None:
    """
    Validate data loading functionality with test cases.
    
    This function can be used to test the data loader without actual Excel files.
    """
    print("=== Data Loader Validation ===")
    
    # Test safe_excel_to_datetime function
    print("\n1. Testing safe_excel_to_datetime:")
    
    test_cases = [
        (44927, "2023-01-01"),  # Excel serial number
        (None, "None"),
        ("2023-01-01T00:00:00.000Z", "2023-01-01"),
        ("invalid", "None"),
        (0.5, "1899-12-30"),  # Half day from Excel epoch
    ]
    
    for input_val, expected_desc in test_cases:
        result = safe_excel_to_datetime(input_val)
        result_str = str(result)[:10] if result else "None"
        print(f"  Input: {input_val} -> {result_str} (Expected: {expected_desc})")
    
    # Test data directory check
    print("\n2. Testing data directory structure:")
    
    current_dir = Path.cwd()
    print(f"  Current working directory: {current_dir}")
    
    # Check if we're running from the data directory itself
    if current_dir.name == "data":
        # Running from data directory, look for Excel files in current directory
        data_dir = current_dir
        print(f"  Running from data directory")
    else:
        # Running from project root or elsewhere, look for data subdirectory
        data_dir = Path("data")
        print(f"  Looking for data subdirectory")
    
    abs_data_dir = data_dir.resolve()
    
    if data_dir.exists():
        print(f"  ✓ Data directory exists: {abs_data_dir}")
        excel_files = list(data_dir.glob("*.xlsx"))
        print(f"  Excel files found: {len(excel_files)}")
        for file in excel_files:
            print(f"    - {file.name}")
    else:
        print(f"  ⚠ Data directory not found: {abs_data_dir}")
    
    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_data_loading()
    
    # Try to load actual data if available
    print("\n=== Attempting Data Load ===")
    if PANDAS_AVAILABLE:
        try:
            container = load_all_market_data()  # Use auto-detection
            print("✓ Data loading successful!")
            validation = container.validate()
            print("Validation results:", validation)
        except Exception as e:
            print(f"⚠ Data loading failed (expected if no Excel files): {e}")
    else:
        print("⚠ Pandas not available - skipping actual data loading test")