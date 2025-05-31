# ライセンス分析レポート - reki-gao プロジェクト

## プロジェクトライセンス
- **プロジェクト**: MIT License (Copyright 2025 karaage)

## 依存関係ライセンス

### 主要依存関係
| パッケージ | バージョン | ライセンス | 互換性 |
|-----------|-----------|-----------|--------|
| fastapi | 0.104.1 | MIT License | ✅ 互換 |
| uvicorn | 0.24.0 | BSD License | ✅ 互換 |
| opencv-python | 4.10.0.84 | Apache 2.0 | ✅ 互換 |
| scikit-learn | 1.4.2 | New BSD | ✅ 互換 |
| numpy | 1.26.4 | BSD-3-Clause | ✅ 互換 |
| pandas | 2.1.1 | BSD-3-Clause | ✅ 互換 |
| pillow | 10.4.0 | HPND | ✅ 互換 |
| pydantic | 2.11.5 | MIT License | ✅ 互換 |
| pydantic-settings | 2.9.1 | MIT License | ✅ 互換 |

### ライセンス詳細

#### MIT License
- **適用**: fastapi, pydantic, pydantic-settings, プロジェクト本体
- **特徴**: 非常に寛容なライセンス、商用利用可能、再配布時に著作権表示が必要

#### BSD License (New BSD / BSD-3-Clause)
- **適用**: uvicorn, scikit-learn, numpy, pandas
- **特徴**: 寛容なライセンス、商用利用可能、再配布時に著作権表示が必要

#### Apache 2.0
- **適用**: opencv-python
- **特徴**: 寛容なライセンス、商用利用可能、特許権の明示的な許諾

#### HPND (Historical Permission Notice and Disclaimer)
- **適用**: pillow
- **特徴**: 非常に寛容なライセンス、商用利用可能

## データソースライセンス

### ROIS-CODH 顔貌コレクション
- **ライセンス**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **提供元**: 人文学オープンデータ共同利用機構（ROIS-CODH）
- **特徴**: 
  - 商用利用可能
  - 改変可能
  - 再配布可能
  - **要件**: 適切なクレジット表示が必要

### クレジット表示例
```
ROIS-CODH「顔貌コレクション」
https://codh.rois.ac.jp/face/
CC BY 4.0
```

## ライセンス互換性分析

### ✅ 問題なし
1. **すべての依存関係**: MIT Licenseと互換性があります
2. **データソース**: CC BY 4.0は適切なクレジット表示により利用可能
3. **商用利用**: すべてのライセンスが商用利用を許可

### ⚠️ 注意事項
1. **クレジット表示**: ROIS-CODHデータ使用時は適切なクレジット表示が必要
2. **再配布**: 各ライセンスの著作権表示要件を満たす必要
3. **特許**: Apache 2.0ライセンス（OpenCV）は特許権の明示的な許諾を含む

## 推奨事項

### 1. クレジット表示の実装
アプリケーション内でデータソースのクレジットを適切に表示：
```json
{
  "credit": "ROIS-CODH「顔貌コレクション」",
  "license": "CC BY 4.0",
  "license_url": "https://creativecommons.org/licenses/by/4.0/"
}
```

### 2. ライセンス情報の文書化
- README.mdにライセンス情報を記載
- 依存関係のライセンス一覧を維持
- データソースのクレジット要件を明記

### 3. 配布時の注意
- すべての著作権表示を保持
- ライセンステキストを含める
- データソースのクレジットを表示

## 結論

**✅ ライセンス的に問題ありません**

- すべての依存関係がMIT Licenseと互換
- データソース（ROIS-CODH）も適切なクレジット表示により利用可能
- 商用利用も含めて法的な問題はありません

ただし、適切なクレジット表示とライセンス情報の維持が重要です。