# .claude/data_schemas/pattern_analysis_schemas.md
# パターン分析データスキーマ

## 概要

`analysis/pattern_analysis/` ディレクトリ内の4つのファイル間で受け渡しされるデータ構造を定義します。パターン一致ラグ相関分析の核心部分であり、データの整合性が予測精度に直結します。

## データフロー概要

```
ProcessedDataContainer → direction_converter.py → DirectionPatterns
                                                       ↓
                        PatternMatches ← pattern_matcher.py
                             ↓
                        optimal_lag_finder.py → OptimalLagResult
                             ↓
                        multi_period_analyzer.py → MultiPeriodPatternResult
```

## 1. direction_converter.py

### 入力データ仕様

```python
from common_types import ProcessedDataContainer

def convert_to_direction_patterns(
    data: ProcessedDataContainer,
    conversion_params: Dict[str, Any]
) -> DirectionPatterns:
    """
    価格データを方向性パターンに変換
    
    Args:
        data: 前処理済みデータ
        conversion_params: 変換パラメータ
    """
    pass

# conversion_params の構造
CONVERSION_PARAMS = {
    'strength_threshold': 0.02,          # 方向性判定閾値 (2%変動)
    'volatility_window': 20,             # ボラティリティ計算窓
    'trend_min_duration': 2,             # 最小トレンド継続日数
    'pattern_min_length': 3,             # 最小パターン長
    'pattern_max_length': 10,            # 最大パターン長
    'smoothing_window': 3,               # 平滑化窓
}
```

### 出力データ仕様

```python
@dataclass
class DirectionPatterns:
    """方向性パターン分析結果"""
    
    btc_directions: pd.DataFrame
    """
    BTC方向性データ
    
    index: pd.DatetimeIndex (日次)
    columns:
        - 'direction': int {-1: 下降, 0: 横ばい, 1: 上昇}
        - 'strength': float [0.0-1.0] (方向性の強さ)
        - 'volatility': float [0.0-1.0] (正規化済みボラティリティ)
        - 'trend_duration': int (同方向継続日数)
        
    制約条件:
        - direction は {-1, 0, 1} のみ
        - strength, volatility は [0.0-1.0] 範囲
        - trend_duration >= 1
        - 欠損値なし
    """
    
    mstr_directions: pd.DataFrame
    """
    MSTR方向性データ (btc_directions と同じ列構成)
    """
    
    btc_pattern_sequences: pd.DataFrame
    """
    BTC高次パターン (連続する方向性の組み合わせ)
    
    index: pd.DatetimeIndex (パターン終了日)
    columns:
        - 'pattern_length': int (パターンの日数: 3-10日)
        - 'pattern_code': str (例: "110", "-1-10", "001")
        - 'pattern_strength': float [0.0-1.0] (パターンの明確さ)
        - 'start_date': pd.Timestamp (パターン開始日)
        
    制約条件:
        - pattern_length は [3, 10] 範囲
        - pattern_code は方向性(-1,0,1)の文字列連結
        - start_date <= index (終了日)
        - パターン重複なし
    """
    
    mstr_pattern_sequences: pd.DataFrame
    """
    MSTR高次パターン (btc_pattern_sequences と同じ列構成)
    """
    
    conversion_params: Dict[str, Any]
    """
    変換時に使用したパラメータ (再現性確保)
    """
    
    quality_metrics: Dict[str, float]
    """
    変換品質指標
    {
        'btc_pattern_coverage': float [0.0-1.0],    # パターン検出カバレッジ
        'mstr_pattern_coverage': float [0.0-1.0],
        'avg_pattern_strength': float [0.0-1.0],    # 平均パターン強度
        'data_completeness': float [0.0-1.0],       # データ完全性
        'direction_consistency': float [0.0-1.0],   # 方向性一貫性
    }
    """
    
    # 検証メソッド
    def validate(self) -> bool:
        """データ整合性検証"""
        checks = [
            self._validate_directions(self.btc_directions),
            self._validate_directions(self.mstr_directions),
            self._validate_patterns(self.btc_pattern_sequences),
            self._validate_patterns(self.mstr_pattern_sequences),
            self._validate_quality_metrics()
        ]
        return all(checks)
```

## 2. pattern_matcher.py

### 入力データ仕様

```python
def find_pattern_matches(
    patterns: DirectionPatterns,
    matching_params: Dict[str, Any]
) -> PatternMatches:
    """
    パターン類似度分析を実行
    
    Args:
        patterns: direction_converter.py の出力
        matching_params: マッチングパラメータ
    """
    pass

# matching_params の構造
MATCHING_PARAMS = {
    'similarity_threshold': 0.7,         # 類似度閾値
    'matching_algorithm': 'cosine',      # アルゴリズム名
    'normalization_method': 'minmax',    # 正規化手法
    'window_size': 30,                   # 比較窓サイズ
    'allow_inverse_patterns': True,     # 逆パターン許可
}
```

### 出力データ仕様

```python
@dataclass
class PatternMatches:
    """パターンマッチング分析結果"""
    
    similarity_matrix: pd.DataFrame
    """
    パターン類似度行列
    
    index: pd.DatetimeIndex (BTCパターン日付)
    columns: pd.DatetimeIndex (MSTRパターン日付)
    values: float [0.0-1.0] (類似度スコア)
    
    制約条件:
        - 値は [0.0-1.0] 範囲
        - 対角線要素は自己相関 (通常 1.0)
        - 上三角行列のみ有効値 (効率化)
        - NaN は未計算を意味
    """
    
    significant_matches: pd.DataFrame
    """
    高類似度ペアの詳細
    
    index: RangeIndex
    columns:
        - 'btc_date': pd.Timestamp (BTCパターン日付)
        - 'mstr_date': pd.Timestamp (MSTRパターン日付)
        - 'similarity_score': float [0.0-1.0]
        - 'btc_pattern_code': str (BTCパターンコード)
        - 'mstr_pattern_code': str (MSTRパターンコード)
        - 'pattern_length': int (パターン長)
        - 'time_lag': int (日数差、正値=MSTR遅延)
        - 'match_type': str {'exact', 'similar', 'inverse'}
    
    制約条件:
        - similarity_score >= similarity_threshold
        - ソート順: similarity_score 降順
        - time_lag = (mstr_date - btc_date).days
        - 重複ペアなし
    """
    
    pattern_statistics: Dict[str, pd.DataFrame]
    """
    パターン統計情報
    {
        'btc_pattern_freq': pd.DataFrame(
            index: パターンコード (str),
            columns: ['frequency', 'avg_strength', 'success_rate']
        ),
        'mstr_pattern_freq': pd.DataFrame(
            # 同じ構成
        ),
        'cross_pattern_freq': pd.DataFrame(
            index: パターンペア (str, 例: "110_-101"),
            columns: ['frequency', 'avg_similarity', 'avg_lag']
        )
    }
    """
    
    matching_quality: Dict[str, float]
    """
    マッチング品質指標
    {
        'total_matches': int,              # 総マッチ数
        'high_quality_matches': int,       # 高品質マッチ数 (score >= 0.8)
        'avg_similarity': float,           # 平均類似度
        'pattern_diversity': float,        # パターン多様性指標
        'temporal_distribution': float,    # 時間分布の均一性
        'coverage_ratio': float,           # カバレッジ比率
    }
    """
    
    matching_params: Dict[str, Any]
    """使用したマッチングパラメータ"""
```

## 3. optimal_lag_finder.py

### 入力データ仕様

```python
def find_optimal_lags(
    patterns: DirectionPatterns,
    matches: PatternMatches,
    lag_params: Dict[str, Any]
) -> OptimalLagResult:
    """
    最適ラグ分析を実行
    
    Args:
        patterns: direction_converter.py の出力
        matches: pattern_matcher.py の出力
        lag_params: ラグ分析パラメータ
    """
    pass

# lag_params の構造
LAG_PARAMS = {
    'max_lag_days': 30,                  # 最大ラグ日数
    'min_correlation': 0.1,              # 最小相関閾値
    'confidence_level': 0.95,            # 信頼水準
    'rolling_window': 252,               # ローリング分析窓
    'stability_threshold': 0.8,          # 安定性閾値
}
```

### 出力データ仕様

```python
@dataclass
class OptimalLagResult:
    """最適ラグ分析結果"""
    
    optimal_lags_by_period: pd.DataFrame
    """
    期間別最適ラグ
    
    index: ['7d', '14d', '30d', '90d', '180d']
    columns:
        - 'optimal_lag': int (最適ラグ日数、正値=BTC先行)
        - 'lag_confidence': float [0.0-1.0] (ラグ信頼度)
        - 'correlation_strength': float [0.0-1.0] (相関強度)
        - 'sample_size': int (分析サンプル数)
        - 'stability_score': float [0.0-1.0] (ラグ安定性)
        
    制約条件:
        - optimal_lag は [-30, 30] 範囲
        - sample_size >= 30 (統計的有意性)
        - 各期間で最低1つの有効ラグ
    """
    
    lag_correlation_matrix: pd.DataFrame
    """
    ラグ別相関分析
    
    index: range(-30, 31)  # ラグ日数
    columns: ['7d', '14d', '30d', '90d', '180d']
    values: float [-1.0-1.0] (各期間・各ラグでの相関係数)
    
    制約条件:
        - 相関係数は [-1.0, 1.0] 範囲
        - 対称性はなし (ラグにより異なる)
    """
    
    dynamic_lag_evolution: pd.DataFrame
    """
    動的ラグ変化 (時系列)
    
    index: pd.DatetimeIndex (分析基準日、月次または週次)
    columns:
        - 'rolling_lag_7d': int (7日予測用ローリングラグ)
        - 'rolling_lag_14d': int
        - 'rolling_lag_30d': int
        - 'rolling_lag_90d': int
        - 'rolling_lag_180d': int
        - 'market_regime': str {'bull', 'bear', 'sideways'}
        - 'volatility_regime': str {'low', 'medium', 'high'}
        
    制約条件:
        - ローリングラグは [-30, 30] 範囲
        - 時系列の連続性確保
        - レジーム分類の一貫性
    """
    
    lag_analysis_details: Dict[str, Any]
    """
    ラグ分析詳細情報
    {
        'significant_lags': List[int],           # 統計的有意ラグリスト
        'lag_distribution': Dict[int, float],    # ラグ分布 {lag: probability}
        'seasonal_effects': Dict[str, float],    # 季節効果 {month: avg_lag}
        'regime_dependent_lags': Dict[str, Dict[str, int]], # レジーム別ラグ
        'cross_validation_scores': Dict[str, float], # 交差検証スコア
        'lag_persistence': float,                # ラグ持続性指標
    }
    """
    
    statistical_tests: Dict[str, Dict[str, Any]]
    """
    統計的検定結果
    {
        'granger_causality': {
            'btc_to_mstr_pvalue': float,
            'mstr_to_btc_pvalue': float,
            'optimal_lag_order': int,
            'test_statistic': float
        },
        'cointegration_test': {
            'test_statistic': float,
            'p_value': float,
            'cointegrated': bool,
            'critical_values': Dict[str, float]
        },
        'lag_stability_test': {
            'chow_test_pvalue': float,
            'structural_break_dates': List[pd.Timestamp],
            'stability_confirmed': bool
        }
    }
    """
```

## 4. multi_period_analyzer.py

### 入力データ仕様

```python
def analyze_multi_period_patterns(
    patterns: DirectionPatterns,
    matches: PatternMatches,
    lag_result: OptimalLagResult,
    analysis_params: Dict[str, Any]
) -> MultiPeriodPatternResult:
    """
    複数期間パターン分析を実行
    
    Args:
        patterns: direction_converter.py の出力
        matches: pattern_matcher.py の出力
        lag_result: optimal_lag_finder.py の出力
        analysis_params: 分析パラメータ
    """
    pass

# analysis_params の構造
ANALYSIS_PARAMS = {
    'consistency_threshold': 0.7,        # 整合性閾値
    'contradiction_sensitivity': 0.8,    # 矛盾検出感度
    'aggregation_method': 'weighted',    # 集約方法
    'confidence_weighting': True,        # 信頼度重み付け
    'temporal_decay': 0.95,              # 時間減衰係数
}
```

### 出力データ仕様

```python
@dataclass
class MultiPeriodPatternResult:
    """複数期間パターン分析結果"""
    
    consolidated_patterns: pd.DataFrame
    """
    統合パターン分析
    
    index: pd.DatetimeIndex (分析基準日)
    columns:
        - 'dominant_pattern_7d': str (7日期間の主要パターン)
        - 'dominant_pattern_14d': str
        - 'dominant_pattern_30d': str
        - 'dominant_pattern_90d': str
        - 'dominant_pattern_180d': str
        - 'pattern_consistency_score': float [0.0-1.0] (期間間一貫性)
        - 'overall_trend_direction': int {-1, 0, 1}
        - 'trend_strength': float [0.0-1.0]
        - 'confidence_aggregate': float [0.0-1.0] (総合信頼度)
        
    制約条件:
        - パターンコードは direction_converter の定義に準拠
        - overall_trend_direction は {-1, 0, 1} のみ
        - 全スコアは [0.0-1.0] 範囲
    """
    
    period_consistency_matrix: pd.DataFrame
    """
    期間間整合性分析
    
    index: ['7d', '14d', '30d', '90d', '180d']
    columns: ['7d', '14d', '30d', '90d', '180d']
    values: float [0.0-1.0] (期間間パターン整合性スコア)
    
    制約条件:
        - 対角線 = 1.0 (自己整合性)
        - 対称行列 (consistency(A,B) = consistency(B,A))
        - 値は [0.0-1.0] 範囲
    """
    
    contradiction_analysis: pd.DataFrame
    """
    矛盾検出結果
    
    index: pd.DatetimeIndex (矛盾検出日)
    columns:
        - 'contradiction_type': str {'directional', 'magnitude', 'timing'}
        - 'conflicting_periods': str (例: "7d_vs_90d")
        - 'contradiction_severity': float [0.0-1.0]
        - 'affected_confidence': float [-1.0-1.0] (信頼度への影響)
        - 'resolution_recommendation': str (対処推奨事項)
        
    制約条件:
        - contradiction_severity >= analysis_params['contradiction_sensitivity']
        - affected_confidence は負値が多い (信頼度低下)
    """
    
    pattern_predictive_power: Dict[str, pd.DataFrame]
    """
    パターン予測力評価
    {
        '7d': pd.DataFrame(
            index: パターンコード,
            columns: ['success_rate', 'avg_accuracy', 'sample_size', 'sharpe_ratio']
        ),
        '14d': pd.DataFrame(...), # 同様構成
        '30d': pd.DataFrame(...),
        '90d': pd.DataFrame(...),
        '180d': pd.DataFrame(...)
    }
    
    制約条件:
        - success_rate は [0.0-1.0] 範囲
        - sample_size >= 5 (統計的意味のある最小サンプル)
        - sharpe_ratio は実数値
    """
    
    overall_quality_metrics: Dict[str, float]
    """
    総合品質指標
    {
        'pattern_stability_score': float [0.0-1.0],      # パターン安定性
        'cross_period_consistency': float [0.0-1.0],     # 期間間一貫性
        'prediction_reliability': float [0.0-1.0],       # 予測信頼性
        'data_coverage': float [0.0-1.0],               # データカバレッジ
        'temporal_consistency': float [0.0-1.0],        # 時系列一貫性
        'anomaly_ratio': float [0.0-1.0],              # 異常値比率
    }
    """
    
    pattern_features_for_prediction: pd.DataFrame
    """
    予測用パターン特徴量 (最重要出力)
    
    index: pd.DatetimeIndex (日次)
    columns:
        - 'lag_adjusted_btc_signal': float (ラグ調整済みBTCシグナル)
        - 'pattern_strength_7d': float [0.0-1.0]
        - 'pattern_strength_14d': float [0.0-1.0]
        - 'pattern_strength_30d': float [0.0-1.0]
        - 'pattern_strength_90d': float [0.0-1.0]
        - 'pattern_strength_180d': float [0.0-1.0]
        - 'consistency_score': float [0.0-1.0]
        - 'regime_indicator': str {'bull', 'bear', 'transition'}
        - 'volatility_adjusted_signal': float
        - 'confidence_weighted_signal': float
        
    制約条件:
        - 全スコアは [0.0-1.0] 範囲
        - 欠損値なし
        - インデックス連続性確保
        
    注意: この DataFrame が prediction モジュールへの主要入力
    """
```

## データ品質・整合性チェック

### 各段階での必須検証項目

#### direction_converter.py 検証
```python
def validate_direction_patterns(patterns: DirectionPatterns) -> Dict[str, bool]:
    """DirectionPatterns の検証"""
    return {
        'valid_directions': patterns.btc_directions['direction'].isin([-1, 0, 1]).all(),
        'strength_range': validate_numeric_range(patterns.btc_directions['strength'], 0.0, 1.0),
        'pattern_coverage': patterns.quality_metrics['btc_pattern_coverage'] >= 0.8,
        'no_missing_values': not patterns.btc_directions.isnull().any().any(),
        'date_alignment': len(patterns.btc_directions) == len(patterns.mstr_directions)
    }
```

#### pattern_matcher.py 検証
```python
def validate_pattern_matches(matches: PatternMatches) -> Dict[str, bool]:
    """PatternMatches の検証"""
    return {
        'similarity_range': validate_numeric_range(
            matches.similarity_matrix.values.flatten(), 0.0, 1.0, allow_nan=True
        ),
        'significant_matches_quality': len(matches.significant_matches) >= 10,
        'temporal_distribution': matches.matching_quality['temporal_distribution'] >= 0.3,
        'no_duplicate_pairs': not matches.significant_matches[['btc_date', 'mstr_date']].duplicated().any()
    }
```

#### optimal_lag_finder.py 検証
```python
def validate_optimal_lag_result(lag_result: OptimalLagResult) -> Dict[str, bool]:
    """OptimalLagResult の検証"""
    return {
        'lag_range': lag_result.optimal_lags_by_period['optimal_lag'].between(-30, 30).all(),
        'confidence_range': validate_numeric_range(
            lag_result.optimal_lags_by_period['lag_confidence'], 0.0, 1.0
        ),
        'sufficient_samples': (lag_result.optimal_lags_by_period['sample_size'] >= 30).all(),
        'statistical_significance': all(
            test['btc_to_mstr_pvalue'] < 0.05 
            for test in [lag_result.statistical_tests['granger_causality']]
        )
    }
```

#### multi_period_analyzer.py 検証
```python
def validate_multi_period_result(result: MultiPeriodPatternResult) -> Dict[str, bool]:
    """MultiPeriodPatternResult の検証"""
    return {
        'consistency_matrix_symmetric': np.allclose(
            result.period_consistency_matrix.values,
            result.period_consistency_matrix.values.T
        ),
        'consistency_diagonal_ones': np.allclose(
            np.diag(result.period_consistency_matrix.values), 1.0
        ),
        'prediction_features_complete': not result.pattern_features_for_prediction.isnull().any().any(),
        'quality_metrics_range': all(
            0.0 <= v <= 1.0 for v in result.overall_quality_metrics.values()
        )
    }
```

## 使用例・実装ガイド

### 基本的な実装フロー
```python
# 1. データ変換
converter = DirectionConverter(CONVERSION_PARAMS)
patterns = converter.convert_to_direction_patterns(processed_data)

# 2. パターンマッチング
matcher = PatternMatcher(MATCHING_PARAMS)
matches = matcher.find_pattern_matches(patterns)

# 3. ラグ分析
lag_finder = OptimalLagFinder(LAG_PARAMS)
lag_result = lag_finder.find_optimal_lags(patterns, matches)

# 4. 統合分析
analyzer = MultiPeriodAnalyzer(ANALYSIS_PARAMS)
final_result = analyzer.analyze_multi_period_patterns(patterns, matches, lag_result)

# 5. 検証
assert final_result.validate(), "Multi-period analysis validation failed"
```

### エラーハンドリング例
```python
try:
    patterns = converter.convert_to_direction_patterns(data)
    if not validate_direction_patterns(patterns)['pattern_coverage']:
        raise DataQualityError("Insufficient pattern coverage")
        
except DataQualityError as e:
    logger.warning(f"Pattern analysis quality issue: {e}")
    # フォールバック処理または再実行
```

---
**重要**: パターン分析は予測精度の基盤となる重要な処理です。各段階での検証を必ず実施し、品質基準を満たすデータのみを次のモジュールに渡してください。