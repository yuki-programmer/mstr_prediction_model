# .claude/data_schemas/decision_schemas.md
# 投資判断データスキーマ

## 概要

`decision/` ディレクトリ内の投資判断モジュールで受け渡しされるデータ構造を定義します。全予測結果を統合し、最終的な投資シグナルを生成する重要な部分です。

## データフロー概要

```
PeriodPredictionResult[] + 補助分析結果 → decision/base_decision.py → DecisionContext
                                                                            ↓
DecisionContext + FactorAnalysisResult[] → decision/multi_period/*.py → InvestmentDecision[]
                        ↑
            decision/factors/*.py (4要因分析)
```

## 1. decision/base_decision.py

### 入力データ仕様

```python
from prediction_schemas import PeriodPredictionResult
from pattern_analysis_schemas import MultiPeriodPatternResult
from typing import Dict, Any, Optional

@dataclass
class InvestmentDecisionInput:
    """投資判断への統合入力データ"""
    
    period_predictions: Dict[str, PeriodPredictionResult]
    """
    全期間予測結果
    
    キー: '7d', '14d', '30d', '90d', '180d'
    値: 各期間のPeriodPredictionResult
    
    制約条件:
    - 全期間が揃っている
    - 予測日が同一または連続
    - 品質基準を満たす
    """
    
    auxiliary_analysis: Dict[str, Any]
    """
    補助分析結果
    {
        'beta_analysis': BetaAnalysisResult,
        'cycle_analysis': CycleAnalysisResult,
        'consistency_analysis': ConsistencyAnalysisResult,
        'pattern_analysis': MultiPeriodPatternResult,
        'btc_predictions': BTCPredictionResult,
    }
    """
    
    market_context: Dict[str, Any]
    """
    市場コンテキスト情報
    {
        'current_btc_price': float,
        'current_mstr_price': float,
        'market_volatility': float [0.0-1.0],
        'trading_volume': float,
        'market_sentiment': str {'bullish', 'bearish', 'neutral'},
        'regulatory_environment': str {'supportive', 'neutral', 'restrictive'},
        'time_of_analysis': pd.Timestamp,
        'market_hours': bool,               # 取引時間内フラグ
        'days_to_earnings': Optional[int],  # 決算まで日数
    }
    """
    
    investment_settings: Dict[str, Any]
    """
    投資設定・制約
    {
        'risk_tolerance': str {'conservative', 'moderate', 'aggressive'},
        'investment_horizon': str {'short', 'medium', 'long'},
        'position_size_limit': float [0.0-1.0],  # ポートフォリオ比率上限
        'stop_loss_threshold': float [0.0-1.0],  # ストップロス閾値
        'take_profit_threshold': float [0.0-5.0], # 利確閾値
        'diversification_requirements': Dict[str, float],
        'leverage_allowed': bool,
        'max_drawdown_tolerance': float [0.0-1.0],
    }
    """

# デフォルト投資設定
DEFAULT_INVESTMENT_SETTINGS = {
    'conservative': {
        'position_size_limit': 0.05,
        'stop_loss_threshold': 0.10,
        'take_profit_threshold': 1.25,
        'max_drawdown_tolerance': 0.15,
    },
    'moderate': {
        'position_size_limit': 0.10,
        'stop_loss_threshold': 0.15,
        'take_profit_threshold': 1.50,
        'max_drawdown_tolerance': 0.25,
    },
    'aggressive': {
        'position_size_limit': 0.20,
        'stop_loss_threshold': 0.20,
        'take_profit_threshold': 2.00,
        'max_drawdown_tolerance': 0.40,
    }
}
```

### 出力データ仕様

```python
@dataclass
class DecisionContext:
    """投資判断統合コンテキスト"""
    
    integrated_forecast: pd.DataFrame
    """
    統合予測サマリー
    
    index: ['7d', '14d', '30d', '90d', '180d']
    columns:
        - 'target_price': float (期間末予測価格)
        - 'expected_return': float (期待リターン)
        - 'confidence_score': float [0.0-1.0] (統合信頼度)
        - 'risk_score': float [0.0-1.0] (統合リスクスコア)
        - 'consistency_score': float [0.0-1.0] (期間間整合性)
        - 'quality_score': float [0.0-1.0] (予測品質)
        - 'volatility_estimate': float (期間ボラティリティ)
        
    制約条件:
    - target_price は正値
    - スコア類は [0.0-1.0] 範囲
    - 期間が長いほど不確実性増加
    """
    
    factor_influences: Dict[str, pd.DataFrame]
    """
    要因別影響度分析
    
    {
        'technical_factors': pd.DataFrame(
            index: ['7d', '14d', '30d', '90d', '180d'],
            columns: ['momentum', 'trend', 'volatility', 'support_resistance']
            values: float [0.0-1.0] (正規化済み影響度)
        ),
        'fundamental_factors': pd.DataFrame(
            index: ['7d', '14d', '30d', '90d', '180d'],
            columns: ['btc_correlation', 'company_fundamentals', 'market_sentiment']
            values: float [0.0-1.0]
        ),
        'pattern_factors': pd.DataFrame(
            index: ['7d', '14d', '30d', '90d', '180d'],
            columns: ['pattern_strength', 'lag_reliability', 'historical_accuracy']
            values: float [0.0-1.0]
        ),
        'cycle_factors': pd.DataFrame(
            index: ['7d', '14d', '30d', '90d', '180d'],
            columns: ['cycle_position', 'seasonal_effects', 'regime_indicators']
            values: float [0.0-1.0]
        )
    }
    
    制約条件:
    - 全値は [0.0-1.0] 範囲で正規化
    - 各期間で要因の重要度順序が一貫
    """
    
    risk_decomposition: Dict[str, Dict[str, float]]
    """
    期間別リスク分解
    
    {
        '7d': {
            'market_risk': float [0.0-1.0],    # 市場リスク
            'specific_risk': float [0.0-1.0],  # 個別銘柄リスク
            'model_risk': float [0.0-1.0],     # モデルリスク
            'liquidity_risk': float [0.0-1.0], # 流動性リスク
            'correlation_risk': float [0.0-1.0] # 相関リスク
        },
        '14d': {...}, # 同様構成
        '30d': {...},
        '90d': {...},
        '180d': {...}
    }
    
    制約条件:
    - 各期間のリスク合計は 1.0 に正規化
    - market_risk が通常最大要因
    """
    
    warnings_and_contradictions: pd.DataFrame
    """
    警告・矛盾検出結果
    
    index: RangeIndex
    columns:
        - 'warning_type': str {'prediction_conflict', 'low_confidence', 'data_quality', 'market_anomaly'}
        - 'severity': str {'low', 'medium', 'high', 'critical'}
        - 'affected_periods': str (例: "7d,14d" または "all")
        - 'description': str (警告内容の詳細)
        - 'impact_on_confidence': float [-1.0-1.0] (信頼度への影響)
        - 'recommended_action': str (推奨対応)
        - 'detection_timestamp': pd.Timestamp
        
    制約条件:
    - severity に応じた対応優先度
    - critical 警告は投資判断停止を推奨
    """
    
    market_environment_assessment: Dict[str, Any]
    """
    市場環境総合評価
    {
        'overall_market_regime': str {'bull', 'bear', 'transition', 'unknown'},
        'volatility_regime': str {'low', 'normal', 'high', 'extreme'},
        'correlation_regime': str {'normal', 'breakdown', 'heightened'},
        'liquidity_conditions': str {'good', 'normal', 'stressed', 'poor'},
        'sentiment_indicators': {
            'fear_greed_index': float [0.0-100.0],
            'volatility_premium': float,
            'momentum_strength': float [-1.0-1.0],
        },
        'macro_environment': str {'supportive', 'neutral', 'headwinds'},
        'regime_confidence': float [0.0-1.0],
    }
    """
    
    data_quality_assessment: Dict[str, float]
    """
    データ品質総合評価
    {
        'overall_data_quality': float [0.0-1.0],
        'prediction_reliability': float [0.0-1.0],
        'model_stability': float [0.0-1.0],
        'historical_accuracy': float [0.0-1.0],
        'coverage_completeness': float [0.0-1.0],
        'recency_score': float [0.0-1.0],
        'cross_validation_score': float [0.0-1.0],
    }
    
    制約条件:
    - 全指標は [0.0-1.0] 範囲
    - overall_data_quality >= 0.8 で投資判断実行推奨
    """
```

## 2. decision/factors/*.py の標準出力仕様

### 共通インターフェース

```python
@dataclass
class FactorAnalysisResult:
    """要因分析結果の共通インターフェース"""
    
    factor_type: str
    """要因タイプ: 'technical', 'fundamental', 'pattern', 'cycle'"""
    
    period_scores: pd.DataFrame
    """
    期間別要因スコア
    
    index: ['7d', '14d', '30d', '90d', '180d']
    columns: (要因タイプにより異なる、下記参照)
    values: float [0.0-1.0] (正規化済みスコア)
    
    制約条件:
    - 全値は [0.0-1.0] 範囲
    - 欠損値なし
    - 期間別の論理的整合性
    """
    
    overall_factor_score: Dict[str, float]
    """
    総合要因スコア (期間別)
    
    {
        '7d': float [0.0-1.0],
        '14d': float [0.0-1.0],
        '30d': float [0.0-1.0],
        '90d': float [0.0-1.0],
        '180d': float [0.0-1.0]
    }
    
    制約条件:
    - 各スコアは period_scores の重み付き平均
    - [0.0-1.0] 範囲
    """
    
    factor_reliability: Dict[str, float]
    """
    要因信頼性 (期間別)
    
    構造は overall_factor_score と同じ
    値は [0.0-1.0] で信頼性の高さを示す
    """
    
    factor_metadata: Dict[str, Any]
    """
    要因固有メタデータ (詳細は各要因で定義)
    """
    
    # 検証メソッド
    def validate(self) -> bool:
        """要因分析結果の妥当性検証"""
        return (
            all(0.0 <= score <= 1.0 for score in self.overall_factor_score.values()) and
            all(0.0 <= rel <= 1.0 for rel in self.factor_reliability.values()) and
            len(self.period_scores) == 5
        )
```

### technical_factors.py 詳細仕様

```python
# period_scores の columns:
TECHNICAL_COLUMNS = [
    'momentum_score',           # モメンタム指標
    'trend_strength',          # トレンド強度
    'volatility_score',        # ボラティリティ指標
    'support_resistance_score', # サポート・レジスタンス
    'volume_score'             # 出来高指標
]

# factor_metadata の構造:
TECHNICAL_METADATA = {
    'rsi_levels': Dict[str, float],          # RSI水準 (期間別)
    'moving_average_signals': Dict[str, str], # 移動平均シグナル
    'bollinger_band_position': Dict[str, float], # ボリンジャーバンド位置
    'volume_profile': {
        'average_volume': float,
        'volume_trend': str,
        'volume_spike_detected': bool,
    },
    'technical_indicators': {
        'macd_signal': str,
        'stochastic_level': float,
        'williams_r': float,
        'cci_reading': float,
    },
    'support_resistance_levels': {
        'key_support': List[float],
        'key_resistance': List[float],
        'current_level_type': str,
    }
}
```

### fundamental_factors.py 詳細仕様

```python
# period_scores の columns:
FUNDAMENTAL_COLUMNS = [
    'btc_correlation_score',    # BTC相関スコア
    'company_health_score',     # 企業健全性スコア
    'market_sentiment_score',   # 市場センチメントスコア
    'regulatory_score',         # 規制環境スコア
    'valuation_score'          # バリュエーションスコア
]

# factor_metadata の構造:
FUNDAMENTAL_METADATA = {
    'current_beta': float,              # 現在のベータ値
    'correlation_stability': float,     # 相関安定性
    'earnings_impact': {
        'last_earnings_surprise': float,
        'next_earnings_date': pd.Timestamp,
        'analyst_revisions': str,
    },
    'institutional_sentiment': {
        'holdings_change': float,
        'institutional_flow': str,
        'analyst_ratings': Dict[str, int],
    },
    'regulatory_events': [
        {
            'event_type': str,
            'impact_assessment': str,
            'probability': float,
            'timeline': str,
        }
    ],
    'valuation_metrics': {
        'price_to_book': float,
        'price_to_sales': float,
        'relative_valuation': str,
    }
}
```

### pattern_factors.py 詳細仕様

```python
# period_scores の columns:
PATTERN_COLUMNS = [
    'pattern_reliability_score',  # パターン信頼性
    'lag_accuracy_score',        # ラグ精度スコア
    'historical_success_score',   # 過去成功率スコア
    'pattern_consistency_score',  # パターン一貫性スコア
    'regime_adaptation_score'    # レジーム適応スコア
]

# factor_metadata の構造:
PATTERN_METADATA = {
    'dominant_patterns': Dict[str, str],      # 期間別主要パターン
    'pattern_success_rates': Dict[str, float], # パターン別成功率
    'lag_stability': Dict[str, float],        # 期間別ラグ安定性
    'pattern_evolution': {
        'pattern_change_frequency': float,
        'adaptation_speed': float,
        'regime_sensitivity': float,
    },
    'cross_validation_results': {
        'out_of_sample_accuracy': Dict[str, float],
        'rolling_window_performance': Dict[str, float],
    },
    'pattern_features_importance': Dict[str, float]
}
```

### cycle_factors.py 詳細仕様

```python
# period_scores の columns:
CYCLE_COLUMNS = [
    'cycle_position_score',    # サイクル位置スコア
    'seasonal_score',         # 季節性スコア
    'regime_score',          # レジームスコア
    'halving_effect_score',  # ハルビング効果スコア
    'macro_cycle_score'      # マクロサイクルスコア
]

# factor_metadata の構造:
CYCLE_METADATA = {
    'current_cycle_phase': str,           # 現在のサイクル段階
    'days_to_halving': int,              # ハルビングまで日数
    'seasonal_patterns': {
        'monthly_effects': Dict[str, float],
        'quarterly_patterns': Dict[str, float],
        'holiday_effects': Dict[str, float],
    },
    'regime_probabilities': {
        'bull_probability': float,
        'bear_probability': float,
        'transition_probability': float,
    },
    'cycle_indicators': {
        'cycle_maturity': float [0.0-1.0],
        'momentum_phase': str,
        'volatility_phase': str,
    }
}
```

## 3. decision/multi_period/*.py の最終出力仕様

### 統一最終出力

```python
@dataclass
class InvestmentDecision:
    """最終投資判断結果"""
    
    # 基本判断情報
    period: str
    """対象期間: '7d', '14d', '30d', '90d', '180d'"""
    
    decision_date: pd.Timestamp
    """判断実行日時"""
    
    # 投資シグナル
    investment_signal: str
    """
    投資シグナル
    値: 'STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'WEAK_SELL', 'SELL', 'STRONG_SELL'
    """
    
    signal_strength: float
    """
    シグナル強度 [0.0-1.0]
    
    0.0-0.2: 弱い
    0.2-0.4: やや弱い  
    0.4-0.6: 中程度
    0.6-0.8: やや強い
    0.8-1.0: 強い
    """
    
    # 価格目標
    price_targets: Dict[str, float]
    """
    価格目標設定
    {
        'target_price': float,      # 主目標価格
        'stop_loss': float,         # ストップロス価格
        'take_profit_1': float,     # 第1利確目標
        'take_profit_2': float,     # 第2利確目標
        'fair_value': float,        # 理論的適正価格
        'support_level': float,     # サポートレベル
        'resistance_level': float,  # レジスタンスレベル
    }
    
    制約条件:
    - 全価格は正値
    - stop_loss < 現在価格 < target_price (BUYシグナル時)
    - take_profit_1 < take_profit_2
    """
    
    # リスク・リターン指標
    risk_return_metrics: Dict[str, float]
    """
    リスク・リターン分析
    {
        'expected_return': float,           # 期待リターン率
        'risk_adjusted_return': float,      # リスク調整後リターン
        'probability_of_profit': float [0.0-1.0], # 利益確率
        'maximum_loss_risk': float [0.0-1.0],     # 最大損失リスク
        'sharpe_ratio_estimate': float,     # 推定シャープレシオ
        'win_loss_ratio': float,           # 勝率
        'expected_drawdown': float,        # 期待ドローダウン
        'tail_risk': float [0.0-1.0],      # テールリスク
    }
    """
    
    # 要因別寄与度
    factor_contributions: Dict[str, float]
    """
    4要因の判断への寄与度
    {
        'technical_weight': float [0.0-1.0],
        'fundamental_weight': float [0.0-1.0],
        'pattern_weight': float [0.0-1.0],
        'cycle_weight': float [0.0-1.0]
    }
    
    制約条件:
    - 合計 = 1.0 (正規化済み)
    - 期間に応じた重み配分
    """
    
    # 信頼度・品質指標
    confidence_metrics: Dict[str, float]
    """
    判断信頼度指標
    {
        'overall_confidence': float [0.0-1.0], # 総合信頼度
        'prediction_quality': float [0.0-1.0], # 予測品質
        'data_sufficiency': float [0.0-1.0],   # データ十分性
        'model_consensus': float [0.0-1.0],    # モデル間一致度
        'historical_accuracy': float [0.0-1.0], # 過去精度
        'regime_consistency': float [0.0-1.0], # レジーム一貫性
    }
    
    制約条件:
    - 全指標は [0.0-1.0] 範囲
    - overall_confidence >= 0.6 で判断実行推奨
    """
    
    # ポジション推奨
    position_recommendation: Dict[str, Any]
    """
    具体的ポジション推奨
    {
        'recommended_allocation': float [0.0-1.0], # 推奨投資比率
        'position_sizing': str {'small', 'medium', 'large'},
        'entry_strategy': str {'immediate', 'gradual', 'wait_for_dip'},
        'exit_strategy': str {'target_based', 'time_based', 'signal_based'},
        'hedging_recommendation': bool,
        'diversification_note': str,
        'leverage_suggestion': Dict[str, Any],
        'rebalancing_frequency': str,
    }
    """
    
    # リスク警告
    risk_warnings: List[Dict[str, Any]]
    """
    リスク警告リスト
    [
        {
            'warning_type': str,              # 警告種別
            'severity': str ['low', 'medium', 'high', 'critical'],
            'description': str,               # 詳細説明
            'mitigation': str,               # 軽減策
            'probability': float [0.0-1.0],  # 発生確率
            'impact': str,                   # 影響度
        }, ...
    ]
    """
    
    # 判断根拠
    decision_rationale: Dict[str, Any]
    """
    判断根拠の詳細
    {
        'primary_drivers': List[str],           # 主要判断要因
        'supporting_evidence': List[str],       # 支持証拠
        'contradictory_signals': List[str],     # 反対シグナル
        'key_assumptions': List[str],           # 主要前提条件
        'sensitivity_analysis': Dict[str, float], # 感度分析結果
        'scenario_robustness': str,             # シナリオ頑健性
        'expert_system_reasoning': List[str],   # エキスパートシステム推論
    }
    """
    
    # メタデータ
    decision_metadata: Dict[str, Any]
    """
    判断メタデータ
    {
        'algorithm_version': str,
        'computation_time': float,            # 計算時間 (秒)
        'data_vintage': pd.Timestamp,        # データ基準日
        'market_conditions': str,            # 判断時市場状況
        'review_date': pd.Timestamp,         # 見直し推奨日
        'expiry_date': pd.Timestamp,         # 判断有効期限
        'decision_id': str,                  # 判断ID (追跡用)
        'parent_analysis_ids': List[str],    # 関連分析ID
    }
    """
```

## 最終データ統合の整合性チェック

### 全期間一貫性検証

```python
def validate_multi_period_investment_decisions(
    decisions: Dict[str, InvestmentDecision]
) -> Dict[str, bool]:
    """複数期間投資判断の整合性検証"""
    
    periods = ['7d', '14d', '30d', '90d', '180d']
    
    # シグナル一貫性チェック
    signals = [decisions[p].investment_signal for p in periods]
    signal_consistency = check_signal_logical_consistency(signals)
    
    # 価格目標一貫性チェック
    target_prices = [decisions[p].price_targets['target_price'] for p in periods]
    price_consistency = check_price_target_consistency(target_prices)
    
    # 信頼度一貫性チェック
    confidences = [decisions[p].confidence_metrics['overall_confidence'] for p in periods]
    confidence_consistency = check_confidence_degradation(confidences)
    
    return {
        'signal_consistency': signal_consistency,
        'price_target_consistency': price_consistency,
        'confidence_consistency': confidence_consistency,
        'risk_escalation_logical': check_risk_escalation(decisions),
        'factor_weight_evolution': check_factor_weight_evolution(decisions)
    }

def check_signal_logical_consistency(signals: List[str]) -> bool:
    """シグナルの論理的一貫性チェック"""
    # BUY系シグナルが連続しているか
    # 短期と長期で大きな矛盾がないか
    buy_signals = ['STRONG_BUY', 'BUY', 'WEAK_BUY']
    sell_signals = ['STRONG_SELL', 'SELL', 'WEAK_SELL']
    
    buy_count = sum(1 for s in signals if s in buy_signals)
    sell_count = sum(1 for s in signals if s in sell_signals)
    
    # 過半数が同じ方向を向いているか
    return max(buy_count, sell_count) >= len(signals) * 0.6
```

### クロス検証要件

```python
def comprehensive_decision_validation(
    decision_context: DecisionContext,
    factor_results: Dict[str, FactorAnalysisResult],
    final_decisions: Dict[str, InvestmentDecision]
) -> Dict[str, Any]:
    """包括的判断検証"""
    
    return {
        # データ整合性
        'context_factors_alignment': validate_context_factors_alignment(
            decision_context, factor_results
        ),
        'prediction_decision_consistency': validate_prediction_decision_consistency(
            decision_context.integrated_forecast, final_decisions
        ),
        
        # 論理的整合性
        'risk_return_logic': validate_risk_return_logic(final_decisions),
        'factor_contribution_logic': validate_factor_contributions(final_decisions),
        
        # 品質保証
        'confidence_threshold_met': all(
            d.confidence_metrics['overall_confidence'] >= 0.6
            for d in final_decisions.values()
        ),
        'data_quality_sufficient': decision_context.data_quality_assessment['overall_data_quality'] >= 0.8,
        
        # 実務的妥当性
        'price_targets_realistic': validate_price_target_realism(final_decisions),
        'position_sizing_appropriate': validate_position_sizing(final_decisions),
    }
```

## 実装ガイド・使用例

### 基本的な投資判断フロー

```python
# 1. 入力データ準備
decision_input = InvestmentDecisionInput(
    period_predictions=period_predictions,
    auxiliary_analysis=auxiliary_results,
    market_context=current_market_context,
    investment_settings=user_risk_profile
)

# 2. 統合コンテキスト生成
base_decision = BaseDecisionMaker()
decision_context = base_decision.create_decision_context(decision_input)

# 3. 4要因分析実行
factor_analyzers = {
    'technical': TechnicalFactorAnalyzer(),
    'fundamental': FundamentalFactorAnalyzer(),
    'pattern': PatternFactorAnalyzer(),
    'cycle': CycleFactorAnalyzer()
}

factor_results = {}
for name, analyzer in factor_analyzers.items():
    factor_results[name] = analyzer.analyze(decision_context)

# 4. 期間別投資判断
investment_decisions = {}
for period in ['7d', '14d', '30d', '90d', '180d']:
    decision_maker = PeriodDecisionMaker(period)
    
    investment_decisions[period] = decision_maker.make_decision(
        decision_context, factor_results
    )

# 5. 包括的検証
validation_results = comprehensive_decision_validation(
    decision_context, factor_results, investment_decisions
)

assert all(validation_results.values()), "Decision validation failed"
```

### エラーハンドリング・フォールバック

```python
try:
    decision_context = base_decision.create_decision_context(decision_input)
    
    # 品質チェック
    if decision_context.data_quality_assessment['overall_data_quality'] < 0.8:
        raise DataQualityError("Insufficient data quality for investment decision")
    
    # 警告チェック
    critical_warnings = decision_context.warnings_and_contradictions[
        decision_context.warnings_and_contradictions['severity'] == 'critical'
    ]
    
    if len(critical_warnings) > 0:
        raise DecisionAbortedError("Critical warnings detected")
        
except DataQualityError as e:
    logger.warning(f"Data quality issue: {e}")
    # 低信頼度での限定的判断または判断停止
    
except DecisionAbortedError as e:
    logger.error(f"Decision aborted: {e}")
    # 安全サイドでの判断（通常はHOLD推奨）
```

---
**最重要**: 投資判断は実際の資産に影響する重要な出力です。全ての検証を確実に実施し、特に以下の点を厳守してください：

1. **信頼度閾値**: overall_confidence >= 0.6
2. **データ品質**: overall_data_quality >= 0.8  
3. **リスク警告**: critical 警告時は判断停止
4. **期間整合性**: 短期・長期判断の論理的一貫性
5. **免責事項**: 投資助言ではない旨の明記