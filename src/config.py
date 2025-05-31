"""
reki-gao アプリケーション設定
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定クラス"""

    # アプリケーション基本設定
    app_name: str = "reki-gao"
    app_version: str = "1.0.0"
    debug: bool = False

    # API設定
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # ファイル処理設定
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png"]
    upload_dir: str = "temp/uploads"

    # 顔認識設定
    face_confidence_threshold: float = 0.8
    face_vector_dimension: int = 128
    max_faces_per_image: int = 5

    # 検索設定
    similarity_search_k: int = 5
    similarity_threshold: float = 0.6

    # データ設定
    data_dir: str = "data"
    kaokore_images_dir: str = "data/kaokore/kaokore/images_256"
    kaokore_tags_path: str = "data/kaokore/kaokore/original_tags.txt"
    kaokore_max_images: Optional[int] = 100  # KaoKore処理画像数制限（Noneで全画像）

    # 外部API設定
    codh_api_base_url: str = "https://codh.rois.ac.jp/face/api"

    # タイムアウト設定
    face_processing_timeout: int = 30
    api_request_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# グローバル設定インスタンス
settings = Settings()


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    return Path(__file__).parent.parent


def ensure_directories():
    """必要なディレクトリを作成"""
    root = get_project_root()
    directories = [
        root / settings.upload_dir,
        root / settings.data_dir,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
