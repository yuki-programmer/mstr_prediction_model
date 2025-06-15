# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MSTR株価予測システム

## プロジェクト概要

マイクロストラテジー(MSTR)株価を予測する金融分析システム。BTCとの相関パターンを独自アルゴリズムで分析し、5段階の時間軸（7日〜180日）で投資判断を支援する。

**核心技術**: パターン一致ラグ相関分析（BTC-MSTR間の時間差を動的に検出）

## 開発環境・ツール

```bash
# 必要な依存関係インストール
pip install -r requirements.txt

# 開発・テスト用コマンド（モジュール単体実行）
python3 data/loader.py            # データローダー単体テスト
python3 data/preprocessor.py      # データ前処理器単体テスト  
python3 config/system_config.py   # システム設定検証
python3 utils/volume_converter.py # ボリューム変換ユーティリティテスト

# 統合テスト（現在利用可能）
python3 test_phase3_integration.py # パターンマッチング統合テスト
python3 analysis/pattern_analysis/optimal_lag_finder.py  # 最適ラグ分析テスト
python3 analysis/pattern_analysis/multi_period_analyzer.py  # Phase2最終統合テスト

# 開発ツール（実装後に利用可能）
python3 -m pytest tests/           # テスト実行
python3 -m pytest tests/test_specific.py::test_function  # 単体テスト実行
python3 -m flake8 .               # リンタ実行
python3 -m black .                # コードフォーマット
python3 -m mypy .                 # 型チェック

# システム実行（実装後）
python3 main.py                   # メイン予測システム実行
```

## アーキテクチャ

### モジュール構成
```
mstr_prediction_system/
├── config/          # システム設定・パラメータ管理
├── data/           # データ処理（Excel読込、前処理、API連携）  
├── analysis/       # パターン分析・相関分析・整合性チェック
├── prediction/     # BTC物理予測・MSTR複数期間予測
├── decision/       # 投資判断・要因分析
├── visualization/  # ダッシュボード・チャート・レポート
└── utils/          # 共通ユーティリティ
```

### データフロー
1. **データ取得**: Excel→ProcessedDataContainer (BTC/MSTR/Gold)
2. **パターン分析**: 方向性変換→パターンマッチング→最適ラグ検出
3. **予測エンジン**: BTC物理予測→MSTR予測（5期間）
4. **投資判断**: 複数予測結果の統合→最終判断

## 重要な設計原則

### データスキーマ準拠
- 実装前に `.claude/data_schemas/` の該当ファイルを必ず確認
- 全DataFrame操作は共通型定義（`common_types.md`）に準拠
- モジュール間データ受け渡しは定義済みスキーマに厳密従う

### コード品質基準
- 型ヒント必須（Type Hints）
- docstring記載必須
- PEP 8準拠
- モジュラー設計（機能独立性）
- 設定駆動（パラメータ外部化）

### 金融ドメイン特性
- **MSTR特性**: ビットコイン大量保有企業の特殊相関
- **ボラティリティ**: 暗号通貨関連銘柄の高変動性
- **リスク管理**: 予測不確実性の明示的考慮
- **免責**: 投資助言ではなく判断支援ツールの位置づけ

## 開発ガイドライン

### 実装順序
1. `data/` - データ基盤構築
2. `analysis/` - 分析エンジン  
3. `prediction/` - 予測モデル
4. `decision/` - 投資判断
5. `visualization/` - 可視化

### 必須参照ドキュメント
- `.claude/context.md` - 詳細仕様・制約条件・ドメイン知識
- `.claude/data_schemas/README.md` - データスキーマ設計全体像
- 各モジュール実装時は対応するスキーマファイルを参照

### エラーハンドリング
- カスタム例外使用：`DataSchemaError`, `DataQualityError`, `DataConsistencyError`
- データ品質チェック必須
- 予測結果の整合性検証

## プロジェクト状態

**現在のフェーズ**: 基盤実装段階
- アーキテクチャ設計完了
- データスキーマ定義完了（`.claude/data_schemas/`）
- 基本モジュール実装済み:
  - `data/loader.py` - Excelデータ読み込み機能
  - `data/preprocessor.py` - データ前処理パイプライン
  - `config/system_config.py` - システム設定管理
  - `utils/volume_converter.py` - ボリューム変換ユーティリティ
  - `analysis/pattern_analysis/direction_converter.py` - GARCH統合型方向性変換
  - `analysis/pattern_analysis/pattern_matcher.py` - 3段階最適化パターンマッチング
  - `analysis/pattern_analysis/optimal_lag_finder.py` - **完全実装**: 階層的最適ラグ分析システム
  - `analysis/pattern_analysis/multi_period_analyzer.py` - **完全実装**: Phase2最終統合モジュール

**Phase2完了**: パターン分析エンジン完成🎉
**次のステップ**: 
1. Phase3予測エンジン（`prediction/`）の開発開始
2. より包括的なテストスイートの作成
3. 実データでの検証とパフォーマンス最適化

## 実装済みモジュール詳細

### データ処理基盤（`data/`）
- **loader.py**: Excel形式の日本語データシート読み込み、RawDataContainer出力
  - 日付の自動変換（Excel serialからTimestamp）、列名の日英マッピング処理
  - BTC/MSTR/Gold各ファイルの統合読み込み、データ整合性検証機能
- **preprocessor.py**: 生データ清浄化、ProcessedDataContainer出力
  - OHLCV制約検証、ボリューム数値変換、データ品質レポート生成

### 設定管理（`config/`）
- **system_config.py**: 包括的設定管理クラス（SystemConfig）
  - 5期間予測パラメータ、BTC-MSTR相関設定、投資リスクプロファイル
  - 環境変数オーバーライド、バリデーション機能、ログ設定

### ユーティリティ（`utils/`）
- **volume_converter.py**: "0.10K"→100.0等のボリューム文字列数値変換

### パターン分析（`analysis/pattern_analysis/`）
- **direction_converter.py**: 高度なGARCH統合型方向性変換モジュール
  - EMA/SMAトレンド分析、GARCHボラティリティ予測、DFAによるHurst指数計算
  - 動的閾値による方向性判定、高次パターンシーケンス生成
  - 包括的品質評価機能、graceful degradation対応
- **pattern_matcher.py**: 3段階最適化パターンマッチングエンジン
  - Phase 1: 基本DTWマッチング、Phase 2: 統計的厳密性、Phase 3: 高度最適化
  - 適応的重み学習、チャンク処理、並列実行サポート
  - ベイジアン信頼度計算、FDR補正、NMS重複除去機能
- **optimal_lag_finder.py**: 完全な階層的最適ラグ分析システム
  - Gold→BTC→MSTR階層的因果構造分析、重み付き最適ラグ計算
  - HMM/閾値ベース動的レジーム検出、時系列ラグ変化分析
  - グランジャー因果性・共和分・構造変化検定、ブートストラップ信頼区間
  - 市場レジーム依存ラグ調整、包括的安定性メトリクス評価
- **multi_period_analyzer.py**: Phase2最終統合・Phase3特徴量生成モジュール
  - 複数期間整合性分析・矛盾検出、ラグ同期信号統合システム
  - 適応的重み付け・動的レジーム分析、市場環境適応機能
  - Phase3向け最適化特徴量生成（15特徴量）、包括的品質評価
  - 統合診断・アンサンブル一致度評価、予測信頼度モデリング

## コア設計パターン

### データフロー制約
- 全DataFrame操作でDatetimeIndex必須、OHLCV列構造厳守
- RawDataContainer→ProcessedDataContainer の変換フロー
- 各モジュールは独立実行可能（__main__でバリデーション関数実行）

### エラーハンドリング戦略
- pandas未インストール時のMock実装によるgraceful degradation
- データ品質不備時の詳細ログ出力とpartial success対応
- 依存関係欠如時の段階的機能制限（warnings経由での通知）

### テスト戦略
- 各モジュールに`__main__`セクションでの独立実行テスト
- `test_phase3_integration.py`による統合テスト（mock/real data対応）
- 依存関係がない環境でもパラメータ検証等の基本テストが実行可能