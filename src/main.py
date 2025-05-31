"""
reki-gao メインエントリーポイント
"""

import asyncio
import logging

from .config import settings, ensure_directories
from .face_detection import FaceDetector
from .face_encoding import FaceEncoder

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def setup_data():
    """データセットアップ（初回実行用）"""
    logger.info("Setting up reki-gao data...")

    # ディレクトリ作成
    ensure_directories()

    # KaoKoreデータセットを使用するため、setup_dataは不要
    # KaoKoreシステムは自動的に初期化される
    logger.info("KaoKore system will be initialized automatically on first API call")
    logger.info("Data setup completed!")


def main():
    """メイン関数"""
    import argparse

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="reki-gao - 歴史的人物顔類似検索システム")
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

    # APIサーバー起動モード
    import uvicorn

    logger.info("Starting reki-gao API server...")
    logger.info(f"KaoKore image limit: {settings.kaokore_max_images or 'All images'}")

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=settings.debug, log_level="info")


if __name__ == "__main__":
    main()
