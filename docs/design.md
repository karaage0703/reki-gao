# 要件・設計書

## 1. 要件定義

### 1.1 基本情報
- ソフトウェア名称: reki-gao
- リポジトリ名: reki-gao

### 1.2 プロジェクト概要

reki-gaoは、現代人の顔写真をもとに、ROIS-CODH「顔貌コレクション（顔コレ）」に含まれる歴史上の人物の顔と似ている人物を見つけて表示するWebアプリケーションです。

**目的・背景:**
- 現代の顔認識技術と歴史的な肖像画データベースを組み合わせ、「時空を超えた顔探し体験」を提供
- 文化的・教育的価値を持つエンターテインメントアプリケーションとして、歴史への興味を促進
- ROIS-CODH「顔貌コレクション」の活用により、日本の歴史的人物との類似性を発見する楽しさを提供

**対象ユーザー:**
- 歴史や文化に興味のある一般ユーザー
- 自分の顔が歴史上の人物に似ているかを知りたいユーザー
- 教育機関での歴史学習ツールとしての利用

### 1.3 機能要件

#### 1.3.1 基本機能
- 顔画像アップロード機能
  - JPEG/PNG形式の画像アップロード対応
  - ドラッグ&ドロップによるファイル選択
  - 画像プレビュー表示
- 顔検出と特徴量抽出機能
  - アップロードされた画像からの顔検出・切り抜き
  - 顔ベクトル（128次元程度）の抽出
  - 複数の顔が検出された場合の処理
- 類似顔検索機能
  - 顔貌コレクション画像との類似度計算
  - コサイン類似度またはL2距離による比較
  - 高速検索のためのベクトルDB活用

#### 1.3.2 結果表示機能
- 類似画像一覧表示
  - 上位3〜5件の類似画像を表示
  - 各人物の画像、名前、時代、出典情報
  - 類似度の可視化（バーグラフ）
- 詳細情報表示
  - 人物の詳細情報（時代、出典、所蔵機関等）
  - ライセンス情報とクレジット表示
- 結果の共有機能（将来実装）

#### 1.3.3 追加機能
- カメラ撮影機能（オプション）
- 和風・古典的なUI/UXデザイン
- レスポンシブデザイン対応

### 1.4 非機能要件

#### 1.4.1 性能要件
- 顔検出・特徴量抽出: 5秒以内
- 類似検索処理: 3秒以内
- 画像アップロード: 10MB以下のファイルサイズ制限
- 同時接続数: 初期段階では10ユーザー程度

#### 1.4.2 セキュリティ要件
- アップロードされた画像の一時保存のみ（永続化しない）
- 個人情報保護への配慮
- HTTPS通信の実装
- 適切なCORSポリシーの設定

#### 1.4.3 運用・保守要件
- ログ記録（エラーログ、アクセスログ）
- 監視機能（基本的なヘルスチェック）
- 定期的なデータベース更新機能

### 1.5 制約条件
- 技術的制約
  - ROIS-CODH「顔貌コレクション」のCC BY 4.0ライセンス遵守
  - 顔認識精度の限界（完全な精度は目指さない）
  - 初期段階はローカル環境での動作を前提
- ビジネス的制約
  - 非商用利用を前提とした設計
  - 教育・文化的価値を重視
- 法的制約
  - 個人情報保護法への準拠
  - 著作権・肖像権への配慮

### 1.6 開発環境
- 言語：Python 3.9+
- バックエンドフレームワーク：FastAPI
- フロントエンド：React/Next.js（将来実装）
- 顔認識ライブラリ：OpenCV, Dlib, FaceNet, DeepFace
- ベクトル検索：FAISS
- 外部API：ROIS-CODH 顔貌コレクション API
- 開発ツール：VSCode
- パッケージ管理：uv

### 1.7 成果物
- ソースコード（バックエンドAPI）
- 設計書
- テストコード
- README（セットアップ手順含む）
- 顔貌コレクションデータ処理スクリプト
- API仕様書

## 2. システム設計

### 2.1 システム概要設計

#### 2.1.1 システムアーキテクチャ
```
[ユーザー] 
    ↓ (画像アップロード)
[FastAPI Backend]
    ↓ (顔検出・特徴量抽出)
[Face Recognition Engine]
    ↓ (ベクトル検索)
[FAISS Vector Database]
    ↓ (メタデータ取得)
[顔貌コレクション Database]
    ↓ (結果返却)
[ユーザー]
```

#### 2.1.2 主要コンポーネント
1. APIサーバー（FastAPI）
   - 画像アップロード処理
   - レスポンス管理
   - エラーハンドリング
2. 顔認識エンジン
   - 顔検出（OpenCV/Dlib）
   - 特徴量抽出（FaceNet）
   - 前処理・後処理
3. 検索エンジン
   - ベクトル類似検索（FAISS）
   - 結果ランキング
   - メタデータ結合
4. データ管理
   - 顔貌コレクションデータ
   - 特徴量ベクトルDB
   - メタデータDB

#### 2.1.3 設定・パラメータ
- 類似検索件数: 5件（デフォルト）
- 顔検出信頼度閾値: 0.8
- 特徴量ベクトル次元数: 128
- 画像サイズ制限: 10MB
- 処理タイムアウト: 30秒

### 2.2 詳細設計

#### 2.2.1 クラス設計

##### 2.2.1.1 FaceDetector
```python
class FaceDetector:
    """顔検出を行うクラス"""

    def __init__(self, confidence_threshold: float = 0.8):
        """初期化メソッド"""
        self.confidence_threshold = confidence_threshold
        self.detector = cv2.dnn.readNetFromTensorflow()

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """画像から顔を検出する"""
        # 顔検出処理
        # 信頼度が閾値以上の顔のみ返却

    def crop_face(self, image: np.ndarray, face_box: Dict) -> np.ndarray:
        """顔領域を切り抜く"""
        # 顔領域の切り抜き処理
```

##### 2.2.1.2 FaceEncoder
```python
class FaceEncoder:
    """顔の特徴量抽出を行うクラス"""

    def __init__(self, model_path: str):
        """初期化メソッド"""
        self.model = self._load_model(model_path)

    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """顔画像から特徴量ベクトルを抽出"""
        # FaceNetモデルによる特徴量抽出

    def _load_model(self, model_path: str):
        """モデルの読み込み"""
        # 事前学習済みモデルの読み込み
```

##### 2.2.1.3 SimilaritySearcher
```python
class SimilaritySearcher:
    """類似顔検索を行うクラス"""

    def __init__(self, index_path: str, metadata_path: str):
        """初期化メソッド"""
        self.index = faiss.read_index(index_path)
        self.metadata = self._load_metadata(metadata_path)

    def search_similar_faces(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """類似顔を検索する"""
        # FAISS による類似検索
        # メタデータと結合して結果を返却

    async def build_index(self, face_vectors: List[np.ndarray]):
        """インデックスを構築する"""
        # FAISSインデックスの構築処理
```

##### 2.2.1.4 GanboCollectionManager
```python
class GanboCollectionManager:
    """顔貌コレクション（ガンボウコレクション）データの管理クラス"""

    def __init__(self, data_dir: str):
        """初期化メソッド"""
        self.data_dir = data_dir
        self.metadata = {}

    async def download_collection_data(self):
        """顔貌コレクションデータをダウンロード"""
        # ROIS-CODH APIからデータ取得

    def preprocess_images(self) -> List[Dict]:
        """画像の前処理と特徴量抽出"""
        # 全画像の顔検出・特徴量抽出

    def get_metadata(self, image_id: str) -> Dict:
        """画像のメタデータを取得"""
        # 人物名、時代、出典等の情報を返却
```

#### 2.2.2 データフロー
1. ユーザーが画像をアップロード
2. FaceDetectorで顔を検出・切り抜き
3. FaceEncoderで特徴量ベクトルを抽出
4. SimilaritySearcherで類似顔を検索
5. GanboCollectionManagerからメタデータを取得
6. 結果をJSON形式でレスポンス

#### 2.2.3 エラーハンドリング
- 顔が検出されない場合のエラー
- 画像フォーマット不正エラー
- ファイルサイズ超過エラー
- 処理タイムアウトエラー
- 外部API接続エラー

### 2.3 インターフェース設計

#### 2.3.1 REST API
- POST /api/v1/upload
  - 画像アップロードと類似顔検索
  - リクエスト: multipart/form-data (image file)
  - レスポンス: JSON (類似顔リスト)
- GET /api/v1/health
  - ヘルスチェック
  - レスポンス: JSON (ステータス情報)
- GET /api/v1/metadata/{image_id}
  - 特定画像のメタデータ取得
  - レスポンス: JSON (詳細情報)

#### 2.3.2 外部連携
- ROIS-CODH 顔貌コレクション API
  - 連携方法: HTTP REST API
  - データフォーマット: JSON
- 顔貌コレクション画像データ
  - 連携方法: HTTP ダウンロード
  - データフォーマット: JPEG画像

### 2.4 セキュリティ設計
- アップロード画像の一時保存のみ（処理後削除）
- ファイルタイプ検証（JPEG/PNG のみ許可）
- ファイルサイズ制限（10MB）
- レート制限の実装
- 適切なCORSヘッダーの設定

### 2.5 テスト設計
- ユニットテスト
  - FaceDetector
    - 顔検出成功ケース
    - 顔検出失敗ケース
    - 複数顔検出ケース
  - FaceEncoder
    - 特徴量抽出成功ケース
    - 無効画像エラーケース
  - SimilaritySearcher
    - 類似検索成功ケース
    - 空結果ケース
- 統合テスト
  - API エンドポイントテスト
  - 画像アップロード〜結果取得の全体フロー
- エラーケーステスト
  - 無効ファイル形式
  - ファイルサイズ超過
  - 顔検出失敗

### 2.6 開発環境・依存関係
- Python 3.9+
- FastAPI
- OpenCV
- dlib
- tensorflow/keras (FaceNet)
- faiss-cpu
- numpy
- Pillow
- uvicorn
- pytest（テスト用）
- uv（パッケージ管理）

### 2.7 開発工程

#### 2.7.1 開発フェーズ
1. 要件分析・定義フェーズ（完了）
   - 要件定義書作成
   - 技術調査
   - アーキテクチャ設計
2. 設計フェーズ（1週間）
   - 詳細設計書作成
   - API仕様設計
   - データベース設計
3. 実装フェーズ（3週間）
   - 顔認識エンジン実装
   - API サーバー実装
   - データ処理スクリプト実装
4. テストフェーズ（1週間）
   - ユニットテスト実装・実行
   - 統合テスト実行
   - 性能テスト実行
5. デプロイ・ドキュメント作成フェーズ（1週間）
   - README作成
   - API仕様書作成
   - デプロイ準備

#### 2.7.2 マイルストーンとタスク優先順位
- マイルストーン1: 基本機能実装完了（2週間後）
  - 顔検出機能実装
  - 特徴量抽出機能実装
  - 基本API実装
- マイルストーン2: 検索機能実装完了（4週間後）
  - FAISS検索エンジン実装
  - 顔貌コレクションデータ処理
  - 類似検索API実装
- マイルストーン3: MVP完成（6週間後）
  - 全機能統合
  - テスト完了
  - ドキュメント完成

#### 2.7.3 リスク管理
- 顔認識精度の問題
  - 対応策: 複数のモデル・手法の検証、閾値調整
- 顔貌コレクションデータの取得困難
  - 対応策: 事前のAPI調査、代替データソースの検討
- 性能要件未達
  - 対応策: 処理の並列化、モデルの軽量化、キャッシュ機能追加

## 3. 運用・保守設計

### 3.1 データセット管理

#### 3.1.1 KaoKoreデータセットのダウンロード

**推奨ダウンロード手順（修正済みスクリプト使用）:**
```bash
# 1. データディレクトリに移動
cd data/kaokore

# 2. KaoKoreリポジトリをクローン
git clone https://github.com/rois-codh/kaokore.git

# 3. 修正済みダウンロードスクリプトを使用
cd kaokore
cp ../../download.py ./download.py  # 修正済みスクリプトをコピー
python download.py
```

**オリジナルスクリプトの問題と修正内容:**

**問題:**
- オリジナルの`download.py`にマルチプロセシング関連のバグが存在
- `pool.close()`と`pool.join()`の呼び出し不足
- 以下のエラーが発生する可能性:
  ```
  BrokenPipeError: [Errno 32] Broken pipe
  ResourceWarning: unclosed <multiprocessing.pool.Pool object>
  ```

**修正内容（GitHub PR #5ベース）:**
```python
# 修正前（オリジナル）
for i, _ in enumerate(pool.imap_unordered(download_and_check_image, zip_params)):
    print("Download images: %7d / %d Done" % (i + 1, len(iurls)), end="\r", flush=True)
print()
# プールのクリーンアップなし → バグの原因

# 修正後（本プロジェクト版）
for i, _ in enumerate(pool.imap_unordered(download_and_check_image, zip_params)):
    print("Download images: %7d / %d Done" % (i + 1, len(iurls)), end="\r", flush=True)
print()

# マルチプロセシングプールの適切なクリーンアップ（GitHub PR #5 修正版）
pool.close()  # 新しいタスクの受付を停止
pool.join()   # 全てのワーカープロセスの完了を待機
```

**注意事項:**
- 約7,500枚の画像ファイルをダウンロード（約1GB）
- ダウンロード時間: 環境により10-30分程度
- マルチプロセシング処理によりCPU使用率が高くなる場合がある
- 修正済みスクリプト使用により安定性が向上

**トラブルシューティング:**
```bash
# エラーが発生する場合の対処法

# 1. シングルスレッドで実行（最も安全）
python download.py --threads 1

# 2. 既存ファイルを上書きして再実行
python download.py --force

# 3. プログレスバー付きで実行
pip install tqdm
python download.py

# 4. オリジナルスクリプトでエラーが出る場合
# 本プロジェクトの修正済みスクリプトを使用
cp ../../download.py ./download.py
```

#### 3.1.2 データ処理の制限設定

**起動時間短縮のための制限:**
- デフォルト設定: 最初の100枚のみ処理
- 処理対象範囲: `00000668.jpg` から `00000767.jpg`
- 起動時間: 約2-3秒（全データの場合は数分）
- メモリ使用量: 約100MB（全データの場合は1GB以上）

**設定変更方法:**

1. **`.env`ファイルでの設定（推奨）:**
```bash
# .envファイルを編集
KAOKORE_MAX_IMAGES=100    # 100枚制限（デフォルト）
KAOKORE_MAX_IMAGES=500    # 500枚制限
KAOKORE_MAX_IMAGES=1000   # 1000枚制限
KAOKORE_MAX_IMAGES=0      # 全画像使用（制限なし）
```

2. **コマンドライン引数での設定:**
```bash
# 起動時に画像数を指定
python -m src.main --max-images 100   # 100枚制限
python -m src.main --max-images 500   # 500枚制限
python -m src.main --max-images 0     # 全画像使用
python -m src.main                    # .env設定を使用
```

3. **設定の優先順位:**
- コマンドライン引数 > `.env`ファイル > デフォルト値(100)

4. **設定ファイルの場所:**
- `.env`: プロジェクトルートの環境設定ファイル
- `src/config.py`: 設定クラスの定義

### 3.2 サーバー運用

#### 3.2.1 起動・停止手順

**正常起動:**
```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# サーバー起動
python -m src.main
```

**ポート競合エラーの対処:**
```bash
# エラー例: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use

# 1. 既存プロセスの確認・終了
lsof -ti:8000 | xargs kill -9

# 2. 別ポートでの起動
uvicorn src.api:app --host 0.0.0.0 --port 8001

# 3. 環境変数での設定
export API_PORT=8001
python -m src.main
```

#### 3.2.2 ログ監視

**ログレベル設定:**
- INFO: 通常運用時の情報
- WARNING: 処理失敗時の警告
- ERROR: システムエラー

**主要ログメッセージ:**
- 起動時: "Starting reki-gao API server..."
- 顔検出: "Detected X faces"
- 検索完了: "Face search completed: X similar faces found"
- エラー: "Face search failed: [エラー詳細]"

### 3.3 性能監視

#### 3.3.1 処理時間の目安

**画像処理時間:**
- 顔検出: 100-500ms
- 特徴量抽出: 50-200ms
- 類似検索: 10-50ms
- 合計: 200-800ms（画像サイズ・品質により変動）

**メモリ使用量:**
- ベースライン: 200-300MB
- 100枚制限時: 400-500MB
- 全データ時: 1.5-2GB

#### 3.3.2 パフォーマンス最適化

**推奨設定:**
- 開発環境: 100枚制限（高速起動）
- 本番環境: 全データ（高精度検索）
- メモリ制約環境: 50枚制限に調整可能

### 3.4 トラブルシューティング

#### 3.4.1 よくある問題と対処法

**1. 起動時のポートエラー**
- 原因: 既存プロセスがポート8000を使用中
- 対処: `lsof -ti:8000 | xargs kill -9` で既存プロセス終了

**2. 顔検出失敗**
- 原因: 画像品質不良、顔が小さすぎる、角度が極端
- 対処: 画像の品質向上、顔検出閾値の調整

**3. メモリ不足**
- 原因: 大量データ処理、メモリリーク
- 対処: 処理枚数制限、定期的な再起動

**4. 検索結果が空**
- 原因: 類似度閾値が高すぎる、データ不足
- 対処: 閾値調整（`similarity_threshold`）、データ範囲拡大

### 3.5 バックアップ・復旧

#### 3.5.1 重要データ

**バックアップ対象:**
- `data/kaokore/` - KaoKoreデータセット
- `src/config.py` - 設定ファイル
- `.env` - 環境変数設定

**復旧手順:**
1. リポジトリの再クローン
2. 依存関係の再インストール
3. KaoKoreデータの再ダウンロード
4. 設定ファイルの復元