# .claude/data_schemas/prediction_schemas.md
# 予測エンジンデータスキーマ

## 概要

`prediction/` ディレクトリ内のBTC予測とMSTR予測モジュール間で受け渡しされるデータ構造を定義します。二段階予測アプローチ（BTC物理的予測 → MSTR予測）の要となる部分です。

## データフロー概要

```
ProcessedDataContainer → btc_prediction/physical_model.py → PhysicalModelResult
                                                                    ↓
                         btc_prediction/btc_predictor.py → BTCPredictionResult
                                                                    ↓
MultiPeriodPatternResult + BTCPredictionResult → multi_period/*.py → PeriodPredictionResult[]
```

## 1. btc_prediction/physical_model.py

### 入力データ仕様

```python
from common_types import ProcessedDataContainer
from typing import Optional

@dataclass
class BTCPhysicalModelInput:
    """BTC物理モデルへの入力データ"""
    
    btc_historical: pd.DataFrame
    """
    BTC履歴データ
    
    必須列構成:
    - index: pd.DatetimeIndex (日付)
    - columns: ['close', 'open', 'high', 'low', 'volume', 'returns']
    
    制約条件:
    - 最低365日以上の履歴データ
    - 欠損値なし
    - 価格データは正値
    """
    
    external_factors: Optional[pd.DataFrame] = None
    """
    外部要因データ (将来拡張用)
    
    列構成:
    - index: pd.DatetimeIndex (btc_historical と対応)
    - columns: ['market_sentiment', 'regulatory_events', 'macro_indicators']
    
    現在は未使用だが、インターフェースとして定義
    """

# 物理モデルパラメータ
PHYSICAL_MODEL_PARAMS = {
    'model_type': 'hybrid',               # モデル種別
    'fitting_period': 365,               # フィッティング期間 (日)
    'validation_split': 0.2,             # 検証用データ比率
    'max_iterations': 1000,              # 最大反復回数
    'convergence_threshold': 1e-6,       # 収束閾値
    'regularization_strength': 0.01,     # 正則化強度
}
```

### 出力データ仕様

```python
@dataclass
class PhysicalModelResult:
    """BTC物理モデルの分析結果"""
    
    model_parameters: Dict[str, float]
    """
    物理モデルパラメータ
    {
        'growth_rate': float,           # 基本成長率 (年率)
        'volatility_decay': float,      # ボラティリティ減衰係数
        'cycle_amplitude': float,       # サイクル振幅
        'cycle_period': float,          # サイクル周期 (日)
        'mean_reversion_speed': float,  # 平均回帰速度
        'long_term_trend': float,       # 長期トレンド係数
        'resistance_levels': List[float], # 抵抗線価格リスト
        'support_levels': List[float],   # 支持線価格リスト
    }
    
    制約条件:
    - growth_rate は実数値 (負値=下落トレンド可)
    - volatility_decay は [0.0, 1.0] 範囲
    - cycle_period は [30, 1500] 日の範囲
    - resistance_levels は昇順ソート
    - support_levels は昇順ソート
    """
    
    physical_indicators: pd.DataFrame
    """
    物理指標時系列 (過去1年分)
    
    index: pd.DatetimeIndex (日次)
    columns:
        - 'momentum_indicator': float (物理的モメンタム)
        - 'energy_level': float [0.0-1.0] (価格エネルギー水準)
        - 'phase_indicator': float [0.0-2π] (サイクル位相)
        - 'stability_measure': float [0.0-1.0] (安定性指標)
        - 'trend_strength': float [-1.0-1.0] (トレンド強度)
        - 'volatility_prediction': float (予測ボラティリティ)
        
    制約条件:
    - energy_level, stability_measure は [0.0-1.0] 範囲
    - phase_indicator は [0, 2π] 範囲
    - trend_strength は [-1.0, 1.0] 範囲
    - volatility_prediction は正値
    """
    
    model_diagnostics: Dict[str, Any]
    """
    モデル診断情報
    {
        'fitting_quality': float [0.0-1.0],    # フィッティング品質
        'residual_analysis': {
            'mean_residual': float,
            'residual_std': float,
            'ljung_box_pvalue': float,          # 系列相関検定
            'jarque_bera_pvalue': float,        # 正規性検定
        },
        'parameter_stability': {
            'growth_rate_std': float,
            'volatility_decay_std': float,
            'cycle_period_std': float,
        },
        'predictive_power': float [0.0-1.0],    # 予測力評価
        'model_complexity': float,              # モデル複雑度
        'overfitting_risk': float [0.0-1.0],   # 過学習リスク
        'computational_time': float,           # 計算時間 (秒)
    }
    """
    
    confidence_parameters: Dict[str, float]
    """
    信頼区間パラメータ
    {
        'base_confidence': float [0.0-1.0],     # 基本信頼度
        'uncertainty_growth_rate': float,       # 不確実性増加率 (日割り)
        'model_uncertainty': float,             # モデル不確実性
        'data_uncertainty': float,              # データ不確実性
        'parameter_uncertainty': float,         # パラメータ不確実性
    }
    
    制約条件:
    - base_confidence は [0.5-1.0] 範囲 (50%以上)
    - uncertainty_growth_rate は正値
    - 各不確実性要因は [0.0-1.0] 範囲
    """
```

## 2. btc_prediction/btc_predictor.py

### 入力データ仕様

```python
def generate_btc_predictions(
    physical_model: PhysicalModelResult,
    btc_data: pd.DataFrame,
    prediction_params: Dict[str, Any]
) -> BTCPredictionResult:
    """
    BTC価格予測を生成
    
    Args:
        physical_model: physical_model.py の出力
        btc_data: ProcessedDataContainer の BTC データ
        prediction_params: 予測パラメータ
    """
    pass

# prediction_params の構造
BTC_PREDICTION_PARAMS = {
    'forecast_periods': ['7d', '14d', '30d', '90d', '180d'],
    'confidence_levels': [0.8, 0.95],    # 信頼区間レベル
    'simulation_count': 10000,           # モンテカルロ回数
    'scenario_count': 3,                 # シナリオ数 (bull/base/bear)
    'volatility_adjustment': True,       # ボラティリティ調整
    'regime_awareness': True,            # レジーム考慮
}
```

### 出力データ仕様

```python
@dataclass
class BTCPredictionResult:
    """BTC価格予測結果"""
    
    predictions_by_period: Dict[str, pd.DataFrame]
    """
    期間別予測価格
    
    キー: '7d', '14d', '30d', '90d', '180d'
    
    各 DataFrame の構成:
    index: pd.DatetimeIndex (将来日付、期間に応じた日数)
    columns:
        - 'predicted_price': float (予測価格、正値)
        - 'confidence_lower_80': float (80%信頼区間下限)
        - 'confidence_upper_80': float (80%信頼区間上限)
        - 'confidence_lower_95': float (95%信頼区間下限)
        - 'confidence_upper_95': float (95%信頼区間上限)
        - 'volatility_forecast': float (予測ボラティリティ、正値)
        - 'trend_component': float (トレンド成分)
        - 'cycle_component': float (サイクル成分)
        - 'noise_component': float (ノイズ成分)
        
    制約条件:
        - predicted_price, confidence bounds は正値
        - confidence_lower <= predicted_price <= confidence_upper
        - volatility_forecast は正値
        - 成分の合計 ≈ predicted_price
    """
    
    prediction_summary: pd.DataFrame
    """
    予測統計サマリー
    
    index: ['7d', '14d', '30d', '90d', '180d']
    columns:
        - 'target_price': float (期間末予測価格)
        - 'expected_return': float (期待リターン率)
        - 'volatility': float (期間ボラティリティ)
        - 'confidence_level': float [0.0-1.0] (予測信頼度)
        - 'upside_potential': float (上値余地 %)
        - 'downside_risk': float (下値リスク %)
        - 'sharpe_estimate': float (推定シャープレシオ)
        
    制約条件:
        - target_price は正値
        - expected_return は実数値
        - volatility は正値
        - confidence_level は [0.0-1.0] 範囲
    """
    
    scenario_analysis: Dict[str, Dict[str, pd.DataFrame]]
    """
    シナリオ分析結果
    
    構造:
    {
        'bull_scenario': {
            '7d': pd.DataFrame,  # predictions_by_period と同じ列構成
            '14d': pd.DataFrame,
            '30d': pd.DataFrame,
            '90d': pd.DataFrame,
            '180d': pd.DataFrame
        },
        'base_scenario': {
            '7d': pd.DataFrame,  # ベースケースシナリオ
            # ... 同様
        },
        'bear_scenario': {
            '7d': pd.DataFrame,  # 弱気シナリオ
            # ... 同様
        }
    }
    
    制約条件:
    - 各シナリオは全期間をカバー
    - bull > base > bear の価格関係
    - 同じ日付範囲とインデックス
    """
    
    accuracy_metrics: Dict[str, float]
    """
    予測精度指標 (過去データでの検証結果)
    {
        'historical_accuracy_7d': float [0.0-1.0],   # 過去7日予測精度
        'historical_accuracy_14d': float [0.0-1.0],
        'historical_accuracy_30d': float [0.0-1.0],
        'historical_accuracy_90d': float [0.0-1.0],
        'historical_accuracy_180d': float [0.0-1.0],
        'mean_absolute_error': float,       # 平均絶対誤差 (%)
        'directional_accuracy': float [0.0-1.0],      # 方向性的中率
        'volatility_forecast_accuracy': float [0.0-1.0], # ボラティリティ予測精度
        'confidence_interval_coverage': float [0.0-1.0], # 信頼区間カバレッジ
    }
    
    制約条件:
    - 精度指標は [0.0-1.0] 範囲
    - mean_absolute_error は正値 (%)
    """
    
    model_integration_info: PhysicalModelResult
    """
    物理モデル統合情報 (参照用)
    physical_model.py の出力をそのまま保持
    """
    
    prediction_metadata: Dict[str, Any]
    """
    予測生成メタデータ
    {
        'generation_timestamp': pd.Timestamp,
        'model_version': str,
        'data_end_date': pd.Timestamp,      # 学習データ終了日
        'prediction_start_date': pd.Timestamp, # 予測開始日
        'prediction_horizon': Dict[str, int],  # 各期間の予測日数
        'market_regime': str,               # 予測時の市場レジーム
        'confidence_adjustment_factor': float, # 信頼度調整係数
        'computational_resources': Dict[str, Any], # 計算リソース情報
    }
    """
```

## 3. multi_period/*.py の統一インターフェース

### 入力データ仕様

```python
@dataclass
class MSTRPredictionInput:
    """MSTR予測への統一入力インターフェース"""
    
    processed_data: ProcessedDataContainer
    """基本データ (common_types.md 参照)"""
    
    btc_predictions: BTCPredictionResult
    """BTC予測結果 (上記で定義)"""
    
    pattern_analysis: MultiPeriodPatternResult
    """パターン分析結果 (pattern_analysis_schemas.md 参照)"""
    
    target_period: str
    """
    予測対象期間
    値: '7d', '14d', '30d', '90d', '180d' のいずれか
    """
    
    prediction_start_date: pd.Timestamp
    """予測開始日 (通常は最新データ日の翌日)"""
    
    auxiliary_analysis: Optional[Dict[str, Any]] = None
    """
    補助分析結果 (オプショナル)
    {
        'beta_analysis': BetaAnalysisResult,
        'cycle_analysis': CycleAnalysisResult,
        'consistency_analysis': ConsistencyAnalysisResult,
    }
    """
    
    prediction_config: Dict[str, Any]
    """
    予測設定
    {
        'use_pattern_weighting': bool,      # パターン重み付け使用
        'btc_influence_factor': float [0.0-1.0], # BTC影響度
        'pattern_influence_factor': float [0.0-1.0], # パターン影響度
        'confidence_threshold': float [0.0-1.0], # 信頼度閾値
        'volatility_adjustment': bool,       # ボラティリティ調整
        'regime_awareness': bool,           # レジーム考慮
        'risk_adjustment': float [0.0-2.0], # リスク調整係数
    }
    
    制約条件:
    - influence_factor の合計は <= 1.0
    - 全フラグは boolean
    - 係数は指定範囲内
    """

# 期間別デフォルト設定
PERIOD_DEFAULT_CONFIGS = {
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
```

### 出力データ仕様

```python
@dataclass
class PeriodPredictionResult:
    """期間別MSTR予測結果の統一出力"""
    
    period: str
    """予測期間 ('7d', '14d', '30d', '90d', '180d')"""
    
    prediction_date: pd.Timestamp
    """予測実行日"""
    
    predicted_prices: pd.Series
    """
    予測価格時系列
    
    index: pd.DatetimeIndex (将来日付)
    values: float (予測MSTR価格、正値)
    
    制約条件:
    - 価格は正値
    - インデックスは連続性確保
    - 期間日数と整合性
    """
    
    confidence_intervals: pd.DataFrame
    """
    信頼区間
    
    index: pd.DatetimeIndex (predicted_prices と同じ)
    columns:
        - 'lower_80': float (80%信頼区間下限)
        - 'upper_80': float (80%信頼区間上限)
        - 'lower_95': float (95%信頼区間下限)
        - 'upper_95': float (95%信頼区間上限)
        
    制約条件:
    - lower <= upper の関係
    - 80% 区間 ⊆ 95% 区間
    - 予測価格が信頼区間内
    """
    
    prediction_factors: Dict[str, pd.Series]
    """
    予測要因分解
    {
        'btc_contribution': pd.Series,      # BTC要因による価格変動寄与
        'pattern_contribution': pd.Series,  # パターン要因寄与
        'trend_contribution': pd.Series,    # トレンド要因寄与
        'volatility_contribution': pd.Series, # ボラティリティ要因寄与
        'residual_contribution': pd.Series, # 残差要因寄与
    }
    
    制約条件:
    - 各 Series の index は predicted_prices と同一
    - 要因の合計 ≈ predicted_prices (分解の整合性)
    - btc_contribution が主要要因 (通常最大)
    """
    
    prediction_statistics: Dict[str, float]
    """
    予測統計サマリー
    {
        'final_price': float,               # 期間末予測価格
        'total_return': float,              # 総リターン率
        'annualized_return': float,         # 年率換算リターン
        'volatility': float,                # 期間ボラティリティ
        'max_drawdown': float,              # 最大ドローダウン
        'upside_probability': float [0.0-1.0], # 上昇確率
        'confidence_score': float [0.0-1.0], # 総合信頼度
        'expected_sharpe': float,           # 期待シャープレシオ
    }
    
    制約条件:
    - final_price は正値
    - リターン率は実数値
    - ボラティリティは正値
    - 確率・信頼度は [0.0-1.0] 範囲
    """
    
    risk_metrics: Dict[str, float]
    """
    リスク指標
    {
        'value_at_risk_95': float,          # VaR (95%水準)
        'conditional_var_95': float,        # CVaR (95%水準)
        'maximum_loss_probability': float [0.0-1.0], # 最大損失確率
        'volatility_risk': float [0.0-1.0], # ボラティリティリスク
        'model_risk': float [0.0-1.0],      # モデルリスク
        'tail_risk': float [0.0-1.0],       # テールリスク
        'correlation_risk': float [0.0-1.0], # 相関リスク
    }
    
    制約条件:
    - VaR, CVaR は負値 (損失額)
    - 確率・リスクは [0.0-1.0] 範囲
    - CVaR <= VaR (条件付き期待損失 >= VaR)
    """
    
    model_diagnostics: Dict[str, Any]
    """
    モデル診断情報
    {
        'btc_correlation_used': float [-1.0-1.0], # 使用したBTC相関
        'pattern_match_quality': float [0.0-1.0], # パターンマッチ品質
        'historical_accuracy': float [0.0-1.0],   # 過去精度
        'regime_consistency': bool,              # レジーム一貫性
        'outlier_adjustments': int,              # 外れ値調整回数
        'convergence_status': str,               # 収束ステータス
        'feature_importance': Dict[str, float],  # 特徴量重要度
        'residual_analysis': Dict[str, float],   # 残差分析結果
    }
    
    制約条件:
    - 相関は [-1.0, 1.0] 範囲
    - 品質・精度は [0.0-1.0] 範囲
    - outlier_adjustments は非負整数
    """
    
    input_references: Dict[str, Any]
    """
    入力データ参照情報
    {
        'btc_prediction_period': str,       # 使用したBTC予測期間
        'pattern_features_used': List[str], # 使用パターン特徴量
        'data_quality_score': float [0.0-1.0], # データ品質スコア
        'effective_sample_size': int,       # 有効サンプルサイズ
        'input_data_end_date': pd.Timestamp, # 入力データ終了日
        'prediction_config_used': Dict[str, Any], # 使用した予測設定
    }
    """
```

## データ連携の整合性チェック

### BTCPredictionResult → PeriodPredictionResult 連携検証

```python
def validate_btc_mstr_prediction_consistency(
    btc_result: BTCPredictionResult,
    mstr_result: PeriodPredictionResult
) -> Dict[str, bool]:
    """BTC予測とMSTR予測の整合性検証"""
    
    period = mstr_result.period
    btc_period_data = btc_result.predictions_by_period.get(period)
    
    if btc_period_data is None:
        return {'period_alignment': False}
    
    return {
        'period_alignment': True,
        'date_range_match': (
            btc_period_data.index[0] == mstr_result.predicted_prices.index[0] and
            btc_period_data.index[-1] == mstr_result.predicted_prices.index[-1]
        ),
        'confidence_consistency': (
            btc_result.prediction_summary.loc[period, 'confidence_level'] <= 
            mstr_result.prediction_statistics['confidence_score']
        ),
        'volatility_relationship': (
            mstr_result.prediction_statistics['volatility'] >= 
            btc_result.prediction_summary.loc[period, 'volatility']
        )
    }
```

### MultiPeriodPatternResult → PeriodPredictionResult 連携検証

```python
def validate_pattern_prediction_consistency(
    pattern_result: MultiPeriodPatternResult,
    mstr_result: PeriodPredictionResult
) -> Dict[str, bool]:
    """パターン分析とMSTR予測の整合性検証"""
    
    period = mstr_result.period
    
    # パターン特徴量の日付整合性
    pattern_features = pattern_result.pattern_features_for_prediction
    prediction_start = mstr_result.predicted_prices.index[0]
    
    return {
        'feature_date_alignment': (
            prediction_start - pd.Timedelta(days=1) in pattern_features.index
        ),
        'confidence_inheritance': (
            pattern_features['confidence_weighted_signal'].iloc[-1] <= 
            mstr_result.prediction_statistics['confidence_score']
        ),
        'pattern_influence_used': (
            'pattern_contribution' in mstr_result.prediction_factors
        ),
        'lag_adjustment_applied': (
            'lag_adjusted_btc_signal' in pattern_features.columns
        )
    }
```

## 実装ガイド・使用例

### 基本的な予測フロー
```python
# 1. BTC物理モデル実行
physical_model = PhysicalModel(PHYSICAL_MODEL_PARAMS)
model_result = physical_model.analyze(btc_data)

# 2. BTC予測生成
btc_predictor = BTCPredictor(BTC_PREDICTION_PARAMS)
btc_predictions = btc_predictor.generate_predictions(model_result, btc_data)

# 3. 各期間のMSTR予測
mstr_predictions = {}
for period in ['7d', '14d', '30d', '90d', '180d']:
    predictor = MSTRPredictor(period, PERIOD_DEFAULT_CONFIGS[period])
    
    input_data = MSTRPredictionInput(
        processed_data=processed_data,
        btc_predictions=btc_predictions,
        pattern_analysis=pattern_result,
        target_period=period,
        prediction_start_date=latest_date + pd.Timedelta(days=1)
    )
    
    mstr_predictions[period] = predictor.predict(input_data)

# 4. 整合性検証
for period, result in mstr_predictions.items():
    consistency = validate_btc_mstr_prediction_consistency(
        btc_predictions, result
    )
    assert all(consistency.values()), f"Consistency check failed for {period}"
```

### エラーハンドリング例
```python
try:
    btc_predictions = btc_predictor.generate_predictions(model_result, btc_data)
    
    # 品質チェック
    if btc_predictions.accuracy_metrics['directional_accuracy'] < 0.6:
        raise DataQualityError("BTC prediction accuracy too low")
        
    # メタデータ検証
    if btc_predictions.prediction_metadata['market_regime'] == 'unknown':
        logger.warning("Market regime detection failed")
        
except DataQualityError as e:
    logger.error(f"BTC prediction quality issue: {e}")
    # フォールバック予測や再実行の実装
```

### パフォーマンス最適化のヒント
```python
# 大容量データの効率的処理
def optimize_prediction_memory(predictions: Dict[str, pd.DataFrame]) -> None:
    """予測データのメモリ最適化"""
    for period_df in predictions.values():
        # float64 → float32 変換 (精度要件に応じて)
        numeric_columns = period_df.select_dtypes(include=['float64']).columns
        period_df[numeric_columns] = period_df[numeric_columns].astype('float32')

# 並列処理での期間別予測
from concurrent.futures import ProcessPoolExecutor

def parallel_mstr_predictions(
    input_data_base: MSTRPredictionInput,
    periods: List[str]
) -> Dict[str, PeriodPredictionResult]:
    """並列実行による高速化"""
    
    with ProcessPoolExecutor(max_workers=len(periods)) as executor:
        futures = {
            period: executor.submit(predict_single_period, input_data_base, period)
            for period in periods
        }
        
        return {
            period: future.result()
            for period, future in futures.items()
        }
```

---
**重要**: BTC予測からMSTR予測への連携は本システムの核心です。データの整合性と品質管理を徹底し、各段階での検証を必ず実施してください。特に信頼区間の妥当性とリスク指標の算出精度は投資判断に直結します。