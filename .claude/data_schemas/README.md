# .claude/data_schemas/README.md
# データスキーマ設計 - インデックス

## 概要

MSTR株価予測システムの全データフローとスキーマ定義を管理するディレクトリです。各モジュール間でやり取りされるデータの形式、型、制約条件を詳細に定義しています。

## ファイル構成

### 📊 [common_types.md](common_types.md)
**全モジュール共通のデータ型定義**
- `ProcessedDataContainer`: 前処理済みデータ
- `RawDataContainer`: 生データコンテナ
- 共通の列構成・インデックス仕様
- データ品質チェック基準

### 🔍 [pattern_analysis_schemas.md](pattern_analysis_schemas.md)
**analysis/pattern_analysis/ の4ファイル間データフロー**
- `direction_converter.py` → `DirectionPatterns`
- `pattern_matcher.py` → `PatternMatches`
- `optimal_lag_finder.py` → `OptimalLagResult`
- `multi_period_analyzer.py` → `MultiPeriodPatternResult`

### 🔮 [prediction_schemas.md](prediction_schemas.md)
**BTC予測とMSTR予測エンジンの連携**
- `btc_prediction/physical_model.py` → `PhysicalModelResult`
- `btc_prediction/btc_predictor.py` → `BTCPredictionResult`
- `multi_period/*.py` → `PeriodPredictionResult`

### 💡 [decision_schemas.md](decision_schemas.md)
**最終投資判断でのデータ統合**
- `decision/base_decision.py` → `DecisionContext`
- `decision/factors/*.py` → `FactorAnalysisResult`
- `decision/multi_period/*.py` → `InvestmentDecision`

## データフロー全体像

```
Excel Files → ProcessedDataContainer → MultiPeriodPatternResult
                                   ↓
                              BTCPredictionResult → PeriodPredictionResult[]
                                                 ↓
                                            DecisionContext → InvestmentDecision[]
```

## 使用方法

### Claude Code での実装時

```bash
# パターン分析モジュール実装
claude "direction_converter.pyを実装。pattern_analysis_schemas.mdの仕様に従って"

# 予測モジュール実装
claude "btc_predictor.pyを実装。prediction_schemas.mdの仕様に従って"

# 投資判断モジュール実装
claude "base_decision.pyを実装。decision_schemas.mdの仕様に従って"
```

### スキーマファイル参照の指針

1. **実装前**: 該当するスキーマファイルで入出力仕様を確認
2. **実装中**: データ型・列構成・制約条件を遵守
3. **テスト時**: スキーマ定義の整合性チェックを実施
4. **更新時**: 関連スキーマファイルの同期更新

## データ品質管理

### 必須チェック項目
- [ ] **型安全性**: 全てのDataFrame/Seriesが定義通りの型
- [ ] **列構成**: 必須列の存在と順序
- [ ] **インデックス**: DatetimeIndexの連続性・重複チェック
- [ ] **値範囲**: 正規化値[0.0-1.0]、確率値等の妥当性
- [ ] **整合性**: モジュール間でのデータ整合性

### パフォーマンス考慮事項
- **メモリ効率**: 大容量DataFrameの適切な型選択
- **計算効率**: ベクトル化演算の活用
- **I/O効率**: 必要最小限のデータ転送

## バージョン管理

### 更新ルール
1. **破壊的変更**: 既存の列名・型変更時はバージョン番号を更新
2. **追加変更**: 新しい列・メタデータ追加時は互換性を保持
3. **ドキュメント**: 変更理由と影響範囲を明記

### 変更履歴
- **v1.0**: 初期スキーマ設計完了
- 今後の変更はここに記録

## トラブルシューティング

### よくある問題
1. **DataFrame列不足**: 必須列の定義確認
2. **型不一致**: pandas dtypeの明示的指定
3. **インデックス不整合**: 日付範囲・頻度の統一
4. **値範囲エラー**: 正規化・スケーリングの確認

### デバッグ支援
- 各スキーマファイルに検証関数の実装例を記載
- `.claude/debug/`にテストデータサンプルを保存

---
**重要**: 実装時は必ず該当するスキーマファイルを参照し、定義された仕様に厳密に従ってください。データの整合性がシステム全体の品質を決定します。