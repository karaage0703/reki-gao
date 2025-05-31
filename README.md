# reki-gao

**reki-gao** は、現代人の顔写真をもとに、ROIS-CODH「顔貌コレクション」（ガンボウコレクション）に含まれる歴史上の人物の顔と似ている人物を見つけて表示する Web アプリケーションです。

## 🎯 概要

現代の顔認識技術と歴史的な肖像画データベースを組み合わせ、「時空を超えた顔探し体験」を提供します。ユーザーがアップロードした顔画像と、古典的な肖像データベース（顔貌コレクション）を比較し、類似した顔を探して表示します。

## ✨ 主な機能

- **顔画像アップロード**: JPEG/PNG形式の画像アップロード対応
- **📷 カメラ撮影機能**: リアルタイムカメラプレビューと写真撮影
- **顔検出・特徴量抽出**: OpenCV LBPベースのハイブリッド特徴量抽出
- **類似顔検索**: コサイン類似度による高速ベクトル検索
- **結果表示**: 類似度と共に歴史上の人物情報を表示
- **WebGUI**: 統合された使いやすいWebインターフェース
- **REST API**: FastAPIベースの高性能API

## 🛠️ 技術スタック

| 項目           | 技術                               |
|----------------|------------------------------------|
| 言語           | Python 3.9+                       |
| Webフレームワーク | FastAPI                          |
| 顔認識         | OpenCV LBP + HOG + 統計的特徴量   |
| ベクトル検索   | scikit-learn NearestNeighbors     |
| 類似度計算     | コサイン類似度                     |
| 画像処理       | OpenCV, Pillow                     |
| フロントエンド | HTML/CSS/JavaScript (WebGUI)      |
| データセット   | ROIS-CODH KaoKore                  |
| テスト         | pytest                             |
| パッケージ管理 | uv                                 |

## 📦 インストール

### 前提条件

- Python 3.9以上
- uv（Pythonパッケージマネージャー）

### uvのインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### プロジェクトのセットアップ

1. **リポジトリのクローン**

```bash
git clone https://github.com/your-username/reki-gao.git
cd reki-gao
```

2. **仮想環境の作成と依存関係のインストール**

```bash
# 仮想環境を作成
uv venv

# 仮想環境をアクティベート
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 依存関係をインストール
uv pip install -r requirements.txt
```

3. **KaoKoreデータセットのダウンロード**

```bash
# KaoKoreデータセットのダウンロード
cd data/kaokore
git clone https://github.com/rois-codh/kaokore.git

# 修正済みダウンロードスクリプトを使用
cd kaokore
cp ../../download.py ./download.py  # 修正済みスクリプトをコピー
python download.py
```

## 🚀 使用方法

### APIサーバーの起動

```bash
# 開発サーバーを起動
python -m src.main

# または直接uvicornで起動
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

サーバーが起動したら、以下のURLでアクセスできます：

- **WebGUI**: http://localhost:8000/ （メインアプリケーション）
- **API ドキュメント**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **ヘルスチェック**: http://localhost:8000/api/v1/health

### API使用例

#### 1. 顔画像をアップロードして類似検索

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_photo.jpg" \
  -F "k=5"
```

#### 2. ヘルスチェック

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

#### 3. 統計情報の取得

```bash
curl -X GET "http://localhost:8000/api/v1/statistics"
```

#### 4. 特定画像のメタデータ取得

```bash
curl -X GET "http://localhost:8000/api/v1/metadata/00000668.jpg"
```

### レスポンス例（顔画像アップロード検索）

```json
{
  "detected_faces": 1,
  "main_face": {
    "confidence": 0.95,
    "method": "opencv"
  },
  "similar_faces": [
    {
      "rank": 1,
      "similarity": 0.85,
      "image_id": "00000701.jpg",
      "person_name": "大橋の中将",
      "era": "平安時代",
      "tags": ["ひき人", "狂人"],
      "gender": "男性",
      "status": "貴族",
      "source": "源氏物語絵巻",
      "collection": "徳川美術館",
      "license": "CC BY 4.0"
    }
  ],
  "search_params": {
    "k": 5,
    "similarity_threshold": 0.6
  }
}
```

## 🧪 テスト

### テストの実行

```bash
# 全テストを実行
pytest

# 詳細な出力でテスト実行
pytest -v

# カバレッジ付きでテスト実行
pytest --cov=src

# 特定のテストファイルのみ実行
pytest tests/test_api.py
```

### テストファイル構成

- `tests/test_face_detection.py` - 顔検出機能のテスト
- `tests/test_face_encoding.py` - 顔特徴量抽出機能のテスト
- `tests/test_api.py` - API エンドポイントのテスト

## 📁 プロジェクト構成

```
reki-gao/
├── src/
│   ├── __init__.py
│   ├── main.py                      # メインエントリーポイント
│   ├── api.py                       # FastAPI アプリケーション
│   ├── config.py                    # 設定管理
│   ├── face_detection.py            # 顔検出機能
│   ├── face_encoding.py             # 顔特徴量抽出機能
│   ├── kaokore_similarity_search.py # KaoKore類似検索機能
│   └── kaokore_loader.py            # KaoKoreデータセット読み込み
├── static/
│   └── index.html                   # WebGUI
├── tests/
│   ├── conftest.py
│   ├── test_face_detection.py
│   ├── test_face_encoding.py
│   └── test_api.py
├── docs/
│   ├── design.md                    # 設計書
│   └── design.md.sample             # 設計書テンプレート
├── data/
│   └── kaokore/                     # KaoKoreデータセット
├── requirements.txt                 # 依存関係
├── README.md
└── LICENSE
```

## ⚙️ 設定

主要な設定は `src/config.py` で管理されています：

```python
# API設定
api_host: str = "0.0.0.0"
api_port: int = 8000

# ファイル処理設定
max_file_size: int = 10 * 1024 * 1024  # 10MB
allowed_extensions: list = [".jpg", ".jpeg", ".png"]

# 顔認識設定
face_confidence_threshold: float = 0.8
face_vector_dimension: int = 128

# 検索設定
similarity_search_k: int = 5
similarity_threshold: float = 0.6
```

環境変数での設定も可能です（`.env` ファイルを作成）：

```bash
DEBUG=true
API_PORT=8080
FACE_CONFIDENCE_THRESHOLD=0.9
```

## 📄 ライセンスとクレジット

### プロジェクトライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

### KaoKoreデータセットについて

顔画像データは ROIS-CODH「KaoKore」（https://github.com/rois-codh/kaokore）を利用しています。

- **ライセンス**: CC BY 4.0
- **クレジット**: ROIS-CODH「KaoKore」
- **URL**: https://github.com/rois-codh/kaokore
- **論文**: "KaoKore: A Pre-modern Japanese Art Facial Expression Dataset"

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 🐛 トラブルシューティング

### よくある問題

#### 1. 依存関係のインストールエラー

```bash
# システムの依存関係が不足している場合
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev libopencv-dev

# macOS
brew install opencv
```

#### 2. 顔検出モデルが見つからない

```bash
# OpenCVのDNNモデルをダウンロード（必要に応じて）
# 実際の実装では自動ダウンロード機能を追加予定
```

#### 3. メモリ不足エラー

- 画像サイズを小さくする
- バッチサイズを調整する
- システムメモリを増やす

#### 4. サーバー起動時のポートエラー

```bash
# エラー: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use

# 既存のプロセスを確認・終了
lsof -ti:8000 | xargs kill -9

# または別のポートで起動
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

#### 5. KaoKoreデータセットのダウンロード

**推奨方法（修正済みスクリプト使用）:**

```bash
# KaoKoreデータセットのダウンロード
cd data/kaokore
git clone https://github.com/rois-codh/kaokore.git

# 修正済みダウンロードスクリプトを使用
cd kaokore
cp ../../download.py ./download.py  # 修正済みスクリプトをコピー
python download.py

# 注意: 大量の画像ファイル（約7,500枚）をダウンロードするため時間がかかります
# ダウンロード完了後、約1GB程度のディスク容量を使用します
```

**オリジナルスクリプトの問題と修正内容:**

オリジナルのKaoKoreダウンロードスクリプトには以下の問題があります：
- マルチプロセシングプールの不適切なクリーンアップ
- `BrokenPipeError`や`ResourceWarning`の発生

本プロジェクトの`data/kaokore/download.py`は**GitHub PR #5の修正版**を適用済みで、以下の修正が含まれています：
```python
# マルチプロセシングプールの適切なクリーンアップ
pool.close()  # 新しいタスクの受付を停止
pool.join()   # 全てのワーカープロセスの完了を待機
```

**エラーが発生する場合の対処法:**

```bash
# シングルスレッドで実行（安全だが低速）
python download.py --threads 1

# 既存ファイルを上書きして再実行
python download.py --force

# プログレスバー付きで実行（tqdmが必要）
pip install tqdm
python download.py
```

#### 6. 起動時間の短縮について

**読み込み枚数の制限**

システムは起動時間を短縮するため、デフォルトでKaoKoreデータセットの最初の100枚のみを処理します：

- 処理対象: `00000668.jpg` から `00000767.jpg` まで
- 起動時間: 約2-3秒（全7,500枚の場合は数分かかる）
- メモリ使用量: 約100MB（全データの場合は1GB以上）

**設定方法:**

1. **`.env`ファイルで設定（推奨）:**
```bash
# .envファイルを編集
KAOKORE_MAX_IMAGES=100    # 100枚制限
KAOKORE_MAX_IMAGES=500    # 500枚制限
KAOKORE_MAX_IMAGES=0      # 全画像使用（制限なし）
```

2. **コマンドライン引数で設定:**
```bash
# 100枚制限で起動
python -m src.main --max-images 100

# 500枚制限で起動
python -m src.main --max-images 500

# 全画像使用で起動
python -m src.main --max-images 0

# .env設定を使用（引数なし）
python -m src.main
```

**注意:** コマンドライン引数は`.env`設定を上書きします。

### ログの確認

```bash
# アプリケーションログの確認
tail -f logs/app.log

# デバッグモードでの実行
DEBUG=true python -m src.main
```

## 📞 サポート

問題や質問がある場合は、以下の方法でお問い合わせください：

- **Issues**: GitHub Issues を作成
- **Discussions**: GitHub Discussions で議論
- **Email**: your-email@example.com

## 🗺️ ロードマップ

- [x] **WebGUI実装** - 統合されたWebインターフェース
- [x] **カメラ撮影機能** - リアルタイムカメラプレビューと写真撮影
- [x] **KaoKoreデータセット統合** - 実際の歴史的人物画像データ
- [x] **特徴量抽出システム** - OpenCV LBPベースのハイブリッド手法
- [x] **統計情報表示** - データセット統計とシステム情報
- [ ] 和風デザインの UI/UX改善
- [ ] 結果の共有機能
- [ ] 多言語対応
- [ ] Docker コンテナ対応
- [ ] クラウドデプロイ対応

---

**reki-gao** - 時空を超えた顔探し体験をお楽しみください！
