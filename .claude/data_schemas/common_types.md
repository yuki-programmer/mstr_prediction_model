# .claude/data_schemas/common_types.md
# 共通データ型定義

## 概要

MSTR予測システム全体で使用される共通のデータ構造、型定義、制約条件を定義します。全モジュールでこれらの定義に準拠することで、データの一貫性と品質を確保します。

## 基本データコンテナ

### RawDataContainer
```python
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

@dataclass
class RawDataContainer:
    """
    Excelファイルから読み込んだ生データコンテナ
    """
    btc_data: pd.DataFrame
    """
    BTC生データ
    
    必須列構成:
    - index: pd.DatetimeIndex (日付)
    - columns: ['date', 'close', 'open', 'high', 'low', 'volume_str', 'change_pct']
    
    データ型:
    - date: datetime64[ns]
    - close, open, high, low: float64
    - volume_str: object (例: "0.10K", "9.38M")
    - change_pct: float64 (小数形式: 0.0154 = 1.54%)
    """
    
    mstr_data: pd.DataFrame
    """
    MSTR生データ (btc_dataと同じ列構成)
    """
    
    gold_data: pd.DataFrame
    """
    Gold生データ (btc_dataと同じ列構成)
    """
    
    data_source: Dict[str, str]
    """
    データソース情報
    {
        'btc_file': 'BTC_USD_daily.xlsx',
        'mstr_file': 'MSTR_daily.xlsx',
        'gold_file': 'gold_daily.xlsx'
    }
    """
    
    load_timestamp: pd.Timestamp
    """データ読み込み実行時刻"""
    
    # 検証メソッド
    def validate(self) -> Dict[str, bool]:
        """データ整合性検証"""
        return {
            'btc_valid': self._validate_dataframe(self.btc_data),
            'mstr_valid': self._validate_dataframe(self.mstr_data),
            'gold_valid': self._validate_dataframe(self.gold_data),
            'date_aligned': self._check_date_alignment()
        }
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """個別DataFrame検証"""
        required_columns = ['date', 'close', 'open', 'high', 'low', 'volume_str', 'change_pct']
        return all(col in df.columns for col in required_columns)
    
    def _check_date_alignment(self) -> bool:
        """日付整合性チェック"""
        # 実装時に詳細化
        return True
```

### ProcessedDataContainer
```python
@dataclass
class ProcessedDataContainer:
    """
    前処理済みデータコンテナ
    """
    btc_processed: pd.DataFrame
    """
    BTC前処理済みデータ
    
    必須列構成:
    - index: pd.DatetimeIndex (日付、連続性確保)
    - columns: ['close', 'open', 'high', 'low', 'volume', 'change_pct', 'returns']
    
    データ型:
    - close, open, high, low: float64 (正値)
    - volume: float64 (K/M変換済み、正値)
    - change_pct: float64 (小数形式)
    - returns: float64 (日次リターン、対数リターン)
    
    制約条件:
    - high >= max(open, close)
    - low <= min(open, close)
    - volume >= 0
    - 欠損値なし
    """
    
    mstr_processed: pd.DataFrame
    """
    MSTR前処理済みデータ (btc_processedと同じ列構成)
    """
    
    gold_processed: pd.DataFrame
    """
    Gold前処理済みデータ (btc_processedと同じ列構成)
    """
    
    common_date_range: tuple[pd.Timestamp, pd.Timestamp]
    """
    共通利用可能期間 
    例: (Timestamp('2020-01-01'), Timestamp('2025-06-05'))
    """
    
    data_quality_report: Dict[str, Any]
    """
    データ品質レポート
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
    前処理パラメータ・ログ
    {
        'volume_conversion_method': str,
        'outlier_detection_method': str,
        'interpolation_method': str,
        'processing_timestamp': pd.Timestamp,
        'data_filters_applied': List[str]
    }
    """
    
    # ユーティリティメソッド
    def get_asset_data(self, asset: str) -> pd.DataFrame:
        """指定資産のデータ取得"""
        asset_map = {
            'btc': self.btc_processed,
            'mstr': self.mstr_processed, 
            'gold': self.gold_processed
        }
        return asset_map.get(asset.lower())
    
    def get_common_period_data(self) -> Dict[str, pd.DataFrame]:
        """共通期間のデータのみ抽出"""
        start, end = self.common_date_range
        return {
            'btc': self.btc_processed.loc[start:end],
            'mstr': self.mstr_processed.loc[start:end],
            'gold': self.gold_processed.loc[start:end]
        }
```

## 共通設定・定数

### 予測期間定義
```python
from enum import Enum
from typing import Final

class PredictionPeriod(Enum):
    """予測期間の標準定義"""
    ULTRA_SHORT = "7d"
    SHORT = "14d" 
    MEDIUM_SHORT = "30d"
    MEDIUM = "90d"
    MEDIUM_LONG = "180d"

# 期間別日数マッピング
PERIOD_DAYS: Final[Dict[str, int]] = {
    "7d": 7,
    "14d": 14,
    "30d": 30,
    "90d": 90,
    "180d": 180
}

# 期間別優先順位 (短期 → 長期)
PERIOD_PRIORITY: Final[List[str]] = ["7d", "14d", "30d", "90d", "180d"]
```

### データ品質基準
```python
# 品質閾値定義
DATA_QUALITY_THRESHOLDS: Final[Dict[str, float]] = {
    'min_data_completeness': 0.95,      # 最低データ完全性 95%
    'max_missing_ratio': 0.05,          # 最大欠損値比率 5%
    'min_quality_score': 0.8,           # 最低品質スコア 80%
    'max_outlier_ratio': 0.02,          # 最大外れ値比率 2%
    'min_confidence_threshold': 0.6,     # 最低信頼度閾値 60%
}

# 正規化範囲
NORMALIZATION_RANGES: Final[Dict[str, tuple[float, float]]] = {
    'confidence_score': (0.0, 1.0),
    'probability': (0.0, 1.0),
    'correlation': (-1.0, 1.0),
    'returns': (-1.0, 1.0),  # 日次リターンの実用範囲
}
```

### 共通列名定義
```python
# 価格データ標準列名
OHLCV_COLUMNS: Final[List[str]] = ['open', 'high', 'low', 'close', 'volume']
PRICE_COLUMNS: Final[List[str]] = ['open', 'high', 'low', 'close']
DERIVED_COLUMNS: Final[List[str]] = ['returns', 'change_pct']

# 日付インデックス名
DATE_INDEX_NAME: Final[str] = 'date'

# 資産名標準化
ASSET_NAMES: Final[Dict[str, str]] = {
    'bitcoin': 'BTC',
    'btc': 'BTC',
    'microstrategy': 'MSTR',
    'mstr': 'MSTR',
    'gold': 'Gold'
}
```

## 共通ユーティリティ関数

### データ検証関数
```python
def validate_dataframe_schema(
    df: pd.DataFrame, 
    required_columns: List[str],
    index_type: type = pd.DatetimeIndex
) -> Dict[str, bool]:
    """
    DataFrame スキーマ検証
    
    Args:
        df: 検証対象DataFrame
        required_columns: 必須列リスト
        index_type: 期待されるインデックス型
        
    Returns:
        検証結果辞書
    """
    return {
        'has_required_columns': all(col in df.columns for col in required_columns),
        'correct_index_type': isinstance(df.index, index_type),
        'no_missing_values': not df.isnull().any().any(),
        'monotonic_index': df.index.is_monotonic_increasing,
        'no_duplicate_index': not df.index.duplicated().any()
    }

def validate_numeric_range(
    series: pd.Series,
    min_val: float = None,
    max_val: float = None,
    allow_nan: bool = False
) -> bool:
    """
    数値系列の範囲検証
    
    Args:
        series: 検証対象Series
        min_val: 最小値 (Noneで無制限)
        max_val: 最大値 (Noneで無制限)
        allow_nan: NaN値許可フラグ
        
    Returns:
        検証結果 (True: 合格)
    """
    if not allow_nan and series.isnull().any():
        return False
    
    clean_series = series.dropna()
    
    if min_val is not None and (clean_series < min_val).any():
        return False
        
    if max_val is not None and (clean_series > max_val).any():
        return False
        
    return True

def check_data_consistency(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    date_tolerance: str = '1D'
) -> Dict[str, Any]:
    """
    複数DataFrame間の整合性チェック
    
    Args:
        df1, df2: 比較対象DataFrame
        date_tolerance: 日付許容差
        
    Returns:
        整合性チェック結果
    """
    return {
        'date_overlap': len(df1.index.intersection(df2.index)),
        'date_gap_days': (df1.index.max() - df2.index.min()).days,
        'frequency_match': df1.index.freq == df2.index.freq,
        'common_period': (
            max(df1.index.min(), df2.index.min()),
            min(df1.index.max(), df2.index.max())
        )
    }
```

### 型変換ユーティリティ
```python
def convert_volume_string(volume_str: str) -> float:
    """
    出来高文字列の数値変換
    
    Args:
        volume_str: "0.10K", "9.38M" 等
        
    Returns:
        変換後の数値
        
    Examples:
        "0.10K" -> 100.0
        "9.38M" -> 9380000.0
        "40.49K" -> 40490.0
    """
    if pd.isna(volume_str) or volume_str == '':
        return 0.0
    
    volume_str = str(volume_str).strip().upper()
    
    try:
        if volume_str.endswith('K'):
            return float(volume_str[:-1]) * 1_000
        elif volume_str.endswith('M'):
            return float(volume_str[:-1]) * 1_000_000
        elif volume_str.endswith('B'):
            return float(volume_str[:-1]) * 1_000_000_000
        else:
            return float(volume_str)
    except (ValueError, TypeError):
        return 0.0

def normalize_to_range(
    series: pd.Series,
    target_range: tuple[float, float] = (0.0, 1.0),
    method: str = 'minmax'
) -> pd.Series:
    """
    系列の正規化
    
    Args:
        series: 正規化対象
        target_range: 目標範囲
        method: 正規化手法 ('minmax', 'zscore', 'robust')
        
    Returns:
        正規化済み系列
    """
    if method == 'minmax':
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)  # 定数系列は中央値
        normalized = (series - min_val) / (max_val - min_val)
    elif method == 'zscore':
        normalized = (series - series.mean()) / series.std()
        # Z-scoreを[0,1]にマッピング (±3σを範囲とする)
        normalized = (normalized + 3) / 6
        normalized = normalized.clip(0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 目標範囲にスケーリング
    target_min, target_max = target_range
    return normalized * (target_max - target_min) + target_min
```

## エラーハンドリング定義

### カスタム例外
```python
class DataSchemaError(Exception):
    """データスキーマ違反エラー"""
    pass

class DataQualityError(Exception):
    """データ品質不足エラー"""
    pass

class DataConsistencyError(Exception):
    """データ整合性エラー"""
    pass

class PredictionPeriodError(Exception):
    """予測期間定義エラー"""
    pass
```

### エラーチェック関数
```python
def assert_schema_compliance(
    df: pd.DataFrame,
    schema_name: str,
    required_columns: List[str]
) -> None:
    """
    スキーマ準拠性の強制チェック
    
    Raises:
        DataSchemaError: スキーマ違反時
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataSchemaError(
            f"{schema_name}: Missing required columns: {missing_columns}"
        )
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataSchemaError(
            f"{schema_name}: Index must be DatetimeIndex, got {type(df.index)}"
        )

def assert_data_quality(
    quality_metrics: Dict[str, float],
    schema_name: str
) -> None:
    """
    データ品質の強制チェック
    
    Raises:
        DataQualityError: 品質基準未達成時
    """
    for metric, threshold in DATA_QUALITY_THRESHOLDS.items():
        if metric in quality_metrics:
            if quality_metrics[metric] < threshold:
                raise DataQualityError(
                    f"{schema_name}: {metric} = {quality_metrics[metric]:.3f} "
                    f"below threshold {threshold:.3f}"
                )
```

---
**重要**: 全モジュールでこれらの共通定義を使用し、データの一貫性を保ってください。新しい共通型が必要な場合は、このファイルに追加してシステム全体で統一してください。