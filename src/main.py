"""
reki-gao メインエントリーポイント
"""

import asyncio
import logging

from .config import settings, ensure_directories
from .ganbo_collection import GanboCollectionManager
from .face_detection import FaceDetector
from .face_encoding import FaceEncoder
from .similarity_search import SimilaritySearcher

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def setup_data():
    """データセットアップ（初回実行用）"""
    logger.info("Setting up reki-gao data...")

    # ディレクトリ作成
    ensure_directories()

    # 顔貌コレクション管理器を初期化
    ganbo_manager = GanboCollectionManager()

    # サンプルデータをダウンロード
    logger.info("Downloading ganbo collection data...")
    await ganbo_manager.download_collection_data(limit=10)

    # 画像を前処理
    processed_images = await ganbo_manager.preprocess_images()
    logger.info(f"Processed {len(processed_images)} images")

    # 顔検出・特徴量抽出の準備
    FaceDetector()  # インスタンス作成（初期化のため）
    FaceEncoder()  # インスタンス作成（初期化のため）
    similarity_searcher = SimilaritySearcher()

    # 各画像から特徴量を抽出してインデックスを構築
    vectors = []
    metadata_list = []

    for image_info in processed_images:
        try:
            # サンプル画像の場合は特徴量をランダム生成
            # 実際の実装では画像から特徴量を抽出
            import numpy as np

            sample_vector = np.random.rand(settings.face_vector_dimension).astype(np.float32)
            sample_vector = sample_vector / np.linalg.norm(sample_vector)  # 正規化

            vectors.append(sample_vector)
            metadata_list.append(image_info["metadata"])

        except Exception as e:
            logger.error(f"Failed to process {image_info['image_id']}: {e}")

    if vectors:
        # インデックスを構築
        logger.info(f"Building search index with {len(vectors)} vectors...")
        similarity_searcher.build_index(vectors, metadata_list)
        similarity_searcher.save_index()
        logger.info("Search index built successfully!")

    logger.info("Data setup completed!")


def main():
    """メイン関数"""
    import argparse

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="reki-gao - 歴史的人物顔類似検索システム")
    parser.add_argument("command", nargs="?", choices=["setup"], help="実行するコマンド")
    parser.add_argument("--max-images", type=int, help="処理する最大画像数（0で全画像、未指定時は.env設定を使用）")
    parser.add_argument("--host", default=settings.api_host, help="APIサーバーのホスト")
    parser.add_argument("--port", type=int, default=settings.api_port, help="APIサーバーのポート")

    args = parser.parse_args()

    # コマンドライン引数で最大画像数が指定された場合、設定を上書き
    if args.max_images is not None:
        max_images = None if args.max_images == 0 else args.max_images
        # 設定を動的に更新
        settings.kaokore_max_images = max_images
        # 既存のインスタンスをリセット（設定変更を反映するため）
        from .kaokore_similarity_search import reset_kaokore_similarity_searcher

        reset_kaokore_similarity_searcher()

    if args.command == "setup":
        # データセットアップモード
        asyncio.run(setup_data())
    else:
        # APIサーバー起動モード
        import uvicorn

        logger.info("Starting reki-gao API server...")
        logger.info(f"KaoKore image limit: {settings.kaokore_max_images or 'All images'}")

        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=settings.debug, log_level="info")


if __name__ == "__main__":
    main()
