"""
FastAPI メインアプリケーション
reki-gao 顔類似検索API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import logging
from typing import List, Dict, Optional
import asyncio
from pathlib import Path
import tempfile
import os

from .config import settings, ensure_directories
from .face_detection import FaceDetector
from .face_encoding import FaceEncoder
from .similarity_search import SimilaritySearcher
from .ganbo_collection import GanboCollectionManager

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション作成
app = FastAPI(
    title="reki-gao API",
    description="現代人の顔写真と歴史上の人物の顔を比較する類似検索API",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数（アプリケーション起動時に初期化）
face_detector: Optional[FaceDetector] = None
face_encoder: Optional[FaceEncoder] = None
similarity_searcher: Optional[SimilaritySearcher] = None
ganbo_manager: Optional[GanboCollectionManager] = None


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化処理"""
    global face_detector, face_encoder, similarity_searcher, ganbo_manager

    logger.info("Starting reki-gao API server...")

    # 必要なディレクトリを作成
    ensure_directories()

    # 各コンポーネントを初期化
    try:
        logger.info("Initializing face detector...")
        face_detector = FaceDetector()

        logger.info("Initializing face encoder...")
        face_encoder = FaceEncoder()

        logger.info("Initializing similarity searcher...")
        similarity_searcher = SimilaritySearcher()

        logger.info("Initializing ganbo collection manager...")
        ganbo_manager = GanboCollectionManager()

        # インデックスが存在しない場合は初期化
        if similarity_searcher.get_index_size() == 0:
            logger.info("No existing index found. Building initial index...")
            await build_initial_index()

        logger.info("reki-gao API server started successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("Shutting down reki-gao API server...")


async def build_initial_index():
    """初期インデックスを構築"""
    try:
        # 顔貌コレクションデータをダウンロード（サンプルデータ）
        logger.info("Downloading ganbo collection data...")
        await ganbo_manager.download_collection_data(limit=10)  # 初期は少数で

        # 画像を前処理
        processed_images = await ganbo_manager.preprocess_images()

        if not processed_images:
            logger.warning("No images found for index building")
            return

        # 各画像から特徴量を抽出
        vectors = []
        metadata_list = []

        for image_info in processed_images:
            try:
                # 画像を読み込み
                image_path = image_info["image_path"]
                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    # サンプル画像を生成（開発用）
                    sample_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
                    cv2.imwrite(image_path, sample_image)

                image = cv2.imread(image_path)
                if image is None:
                    continue

                # 顔検出
                faces = face_detector.detect_faces(image)
                if not faces:
                    # 顔が検出されない場合は画像全体を使用
                    face_crop = image
                else:
                    # 最初の顔を使用
                    face_crop = face_detector.crop_face(image, faces[0])

                if face_crop is None:
                    continue

                # 前処理
                preprocessed_face = face_detector.preprocess_face_for_encoding(face_crop)
                if preprocessed_face is None:
                    continue

                # 特徴量抽出
                encoding = face_encoder.encode_face(preprocessed_face)
                if encoding is not None and face_encoder.is_valid_encoding(encoding):
                    vectors.append(encoding)
                    metadata_list.append(image_info["metadata"])

            except Exception as e:
                logger.error(f"Failed to process image {image_info['image_id']}: {e}")
                continue

        if vectors:
            # インデックスを構築
            logger.info(f"Building index with {len(vectors)} vectors...")
            similarity_searcher.build_index(vectors, metadata_list)
            similarity_searcher.save_index()
            logger.info("Index built successfully!")
        else:
            logger.warning("No valid face encodings found for index building")

    except Exception as e:
        logger.error(f"Failed to build initial index: {e}")


def get_face_detector() -> FaceDetector:
    """顔検出器の依存性注入"""
    if face_detector is None:
        raise HTTPException(status_code=500, detail="Face detector not initialized")
    return face_detector


def get_face_encoder() -> FaceEncoder:
    """顔エンコーダーの依存性注入"""
    if face_encoder is None:
        raise HTTPException(status_code=500, detail="Face encoder not initialized")
    return face_encoder


def get_similarity_searcher() -> SimilaritySearcher:
    """類似検索器の依存性注入"""
    if similarity_searcher is None:
        raise HTTPException(status_code=500, detail="Similarity searcher not initialized")
    return similarity_searcher


def get_ganbo_manager() -> GanboCollectionManager:
    """顔貌コレクション管理器の依存性注入"""
    if ganbo_manager is None:
        raise HTTPException(status_code=500, detail="Ganbo collection manager not initialized")
    return ganbo_manager


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "reki-gao API",
        "description": "現代人の顔写真と歴史上の人物の顔を比較する類似検索API",
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/api/v1/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        # 各コンポーネントの状態をチェック
        detector_status = face_detector is not None
        encoder_status = face_encoder is not None
        searcher_status = similarity_searcher is not None
        manager_status = ganbo_manager is not None

        index_size = similarity_searcher.get_index_size() if similarity_searcher else 0

        return {
            "status": "healthy",
            "components": {
                "face_detector": detector_status,
                "face_encoder": encoder_status,
                "similarity_searcher": searcher_status,
                "ganbo_manager": manager_status,
            },
            "index_size": index_size,
            "version": settings.app_version,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/api/v1/upload")
async def upload_and_search(
    file: UploadFile = File(...),
    k: int = 5,
    detector: FaceDetector = Depends(get_face_detector),
    encoder: FaceEncoder = Depends(get_face_encoder),
    searcher: SimilaritySearcher = Depends(get_similarity_searcher),
):
    """
    画像をアップロードして類似顔検索を実行

    Args:
        file: アップロードする画像ファイル
        k: 返す類似顔の数（デフォルト: 5）

    Returns:
        類似顔検索結果
    """
    try:
        # ファイル形式チェック
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # ファイルサイズチェック
        contents = await file.read()
        if len(contents) > settings.max_file_size:
            raise HTTPException(status_code=400, detail="File size too large")

        # 画像を読み込み
        try:
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image.convert("RGB"))
            # OpenCVはBGRなので変換
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 顔検出
        faces = detector.detect_faces(image_array)
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # 最初の顔を使用（複数検出された場合）
        main_face = faces[0]
        face_crop = detector.crop_face(image_array, main_face)

        if face_crop is None:
            raise HTTPException(status_code=400, detail="Failed to crop face from image")

        # 前処理
        preprocessed_face = detector.preprocess_face_for_encoding(face_crop)
        if preprocessed_face is None:
            raise HTTPException(status_code=400, detail="Failed to preprocess face image")

        # 特徴量抽出
        encoding = encoder.encode_face(preprocessed_face)
        if encoding is None or not encoder.is_valid_encoding(encoding):
            raise HTTPException(status_code=400, detail="Failed to extract face features")

        # 類似検索
        similar_faces = searcher.search_similar_faces(encoding, k=k)

        # 結果を整形
        result = {
            "detected_faces": len(faces),
            "main_face": {"confidence": main_face["confidence"], "method": main_face["method"]},
            "similar_faces": similar_faces,
            "search_params": {"k": k, "similarity_threshold": settings.similarity_threshold},
        }

        logger.info(f"Face search completed: {len(similar_faces)} similar faces found")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during face search")


@app.get("/api/v1/metadata/{image_id}")
async def get_metadata(image_id: str, manager: GanboCollectionManager = Depends(get_ganbo_manager)):
    """
    特定画像のメタデータを取得

    Args:
        image_id: 画像ID

    Returns:
        画像のメタデータ
    """
    try:
        metadata = manager.get_metadata(image_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Image metadata not found")

        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata for {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metadata")


@app.get("/api/v1/statistics")
async def get_statistics(
    searcher: SimilaritySearcher = Depends(get_similarity_searcher),
    manager: GanboCollectionManager = Depends(get_ganbo_manager),
):
    """
    システム統計情報を取得

    Returns:
        統計情報
    """
    try:
        search_stats = searcher.get_statistics()
        collection_stats = manager.get_statistics()

        return {
            "search_engine": search_stats,
            "collection": collection_stats,
            "system": {
                "version": settings.app_version,
                "face_vector_dimension": settings.face_vector_dimension,
                "similarity_threshold": settings.similarity_threshold,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.post("/api/v1/admin/rebuild-index")
async def rebuild_index(
    manager: GanboCollectionManager = Depends(get_ganbo_manager),
    searcher: SimilaritySearcher = Depends(get_similarity_searcher),
):
    """
    インデックスを再構築（管理者用）

    Returns:
        再構築結果
    """
    try:
        logger.info("Starting index rebuild...")
        await build_initial_index()

        new_size = searcher.get_index_size()
        logger.info(f"Index rebuild completed. New size: {new_size}")

        return {"status": "success", "message": "Index rebuilt successfully", "new_index_size": new_size}

    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to rebuild index")


if __name__ == "__main__":
    # 開発用サーバー起動
    uvicorn.run("src.api:app", host=settings.api_host, port=settings.api_port, reload=settings.debug, log_level="info")
