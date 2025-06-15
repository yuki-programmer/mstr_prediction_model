"""
Volume converter utilities for MSTR prediction system.

This module provides functions to convert volume string formats (e.g., "0.10K", "9.38M") 
to numerical values as required by the data processing pipeline.
"""

from typing import Union, Optional

try:
    import pandas as pd
except ImportError:
    # Fallback for testing without pandas
    class MockPandas:
        @staticmethod
        def isna(value):
            return value is None or (hasattr(value, '__len__') and len(str(value).strip()) == 0)
        
        class NA:
            pass
    
    pd = MockPandas()


def convert_volume_string(volume_str: str) -> float:
    """
    Convert volume string format to numerical value.
    
    Handles common volume string formats with suffixes K (thousands), 
    M (millions), and B (billions), converting them to their numerical equivalents.
    
    Args:
        volume_str: Volume string to convert. Examples: "0.10K", "9.38M", "40.49K"
                   Can also be NaN, None, empty string, or plain number string.
    
    Returns:
        Converted numerical value as float. Returns 0.0 for invalid inputs.
        
    Examples:
        >>> convert_volume_string("0.10K")
        100.0
        >>> convert_volume_string("9.38M")
        9380000.0
        >>> convert_volume_string("40.49K")
        40490.0
        >>> convert_volume_string("1.5B")
        1500000000.0
        >>> convert_volume_string("12345")
        12345.0
        >>> convert_volume_string("")
        0.0
        >>> convert_volume_string(None)
        0.0
    """
    # Handle NaN, None, or empty string
    if pd.isna(volume_str) or volume_str is None or volume_str == '':
        return 0.0
    
    # Convert to string and normalize
    volume_str = str(volume_str).strip().upper()
    
    try:
        if volume_str.endswith('K'):
            # Remove 'K' suffix and multiply by 1,000
            return float(volume_str[:-1]) * 1_000
        elif volume_str.endswith('M'):
            # Remove 'M' suffix and multiply by 1,000,000
            return float(volume_str[:-1]) * 1_000_000
        elif volume_str.endswith('B'):
            # Remove 'B' suffix and multiply by 1,000,000,000
            return float(volume_str[:-1]) * 1_000_000_000
        else:
            # No suffix, convert directly to float
            return float(volume_str)
    except (ValueError, TypeError):
        # Return 0.0 for any conversion errors
        return 0.0


def validate_volume_conversion() -> None:
    """
    Validate volume conversion function with various test cases.
    
    Tests both normal and edge cases to ensure proper functionality.
    Prints test results to stdout.
    """
    print("=== Volume Conversion Validation ===")
    
    # Test cases: (input, expected_output)
    test_cases = [
        # Normal cases with suffixes
        ("0.10K", 100.0),
        ("9.38M", 9380000.0),
        ("40.49K", 40490.0),
        ("1.5B", 1500000000.0),
        ("2.75M", 2750000.0),
        ("500K", 500000.0),
        
        # Cases without suffixes
        ("12345", 12345.0),
        ("0", 0.0),
        ("1.23", 1.23),
        
        # Edge cases
        ("", 0.0),
        (None, 0.0),
        ("invalid", 0.0),
        ("K", 0.0),
        ("M", 0.0),
        ("B", 0.0),
        ("1.2.3K", 0.0),
        
        # Case sensitivity tests
        ("1.5k", 1500.0),
        ("2.3m", 2300000.0),
        ("0.5b", 500000000.0),
        
        # Whitespace handling
        (" 1.5K ", 1500.0),
        ("  2.3M  ", 2300000.0),
    ]
    
    # Test with pandas NaN
    try:
        import numpy as np
        nan_test_cases = [
            (np.nan, 0.0),
            (pd.NA, 0.0),
        ]
    except ImportError:
        # Fallback for testing without numpy/pandas
        nan_test_cases = [
            (None, 0.0),
        ]
    
    all_passed = True
    total_tests = len(test_cases) + len(nan_test_cases)
    passed_tests = 0
    
    # Run regular test cases
    for i, (input_val, expected) in enumerate(test_cases, 1):
        try:
            result = convert_volume_string(input_val)
            if abs(result - expected) < 1e-6:  # Float comparison with tolerance
                print(f"✓ Test {i:2d}: '{input_val}' -> {result} (Expected: {expected})")
                passed_tests += 1
            else:
                print(f"✗ Test {i:2d}: '{input_val}' -> {result} (Expected: {expected})")
                all_passed = False
        except Exception as e:
            print(f"✗ Test {i:2d}: '{input_val}' -> ERROR: {e}")
            all_passed = False
    
    # Run NaN test cases
    for i, (input_val, expected) in enumerate(nan_test_cases, len(test_cases) + 1):
        try:
            result = convert_volume_string(input_val)
            if abs(result - expected) < 1e-6:
                print(f"✓ Test {i:2d}: NaN/NA -> {result} (Expected: {expected})")
                passed_tests += 1
            else:
                print(f"✗ Test {i:2d}: NaN/NA -> {result} (Expected: {expected})")
                all_passed = False
        except Exception as e:
            print(f"✗ Test {i:2d}: NaN/NA -> ERROR: {e}")
            all_passed = False
    
    # Summary
    print(f"\n=== Validation Summary ===")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    print(f"Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("Please review failed test cases above.")


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_volume_conversion()