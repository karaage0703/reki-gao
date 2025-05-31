"""
FastAPI メインアプリケーション
reki-gao 顔類似検索API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import logging
from typing import Optional

from .config import settings, ensure_directories
from .face_detection import FaceDetector
from .face_encoding import FaceEncoder
from .kaokore_loader import kaokore_loader
from .kaokore_similarity_search import get_kaokore_similarity_searcher

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

# Staticファイルのマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# グローバル変数（アプリケーション起動時に初期化）
face_detector: Optional[FaceDetector] = None
face_encoder: Optional[FaceEncoder] = None


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化処理"""
    global face_detector, face_encoder

    logger.info("Starting reki-gao API server...")

    # 必要なディレクトリを作成
    ensure_directories()

    # 各コンポーネントを初期化
    try:
        logger.info("Initializing face detector...")
        face_detector = FaceDetector()

        logger.info("Initializing face encoder...")
        face_encoder = FaceEncoder()

        # KaoKore類似検索の初期化（設定ファイルまたはコマンドライン引数から制限値を取得）
        logger.info("Initializing KaoKore similarity searcher...")
        get_kaokore_similarity_searcher()  # インスタンスを初期化（グローバル変数に保存される）

        logger.info("reki-gao API server started successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("Shutting down reki-gao API server...")


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


@app.get("/")
async def root():
    """ルートエンドポイント - GUIにリダイレクト"""
    return FileResponse("static/index.html")


@app.get("/api/v1/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        # 各コンポーネントの状態をチェック
        detector_status = face_detector is not None
        encoder_status = face_encoder is not None

        # KaoKore類似検索の状態をチェック
        kaokore_searcher = get_kaokore_similarity_searcher()
        kaokore_status = kaokore_searcher is not None
        index_size = len(kaokore_searcher.vectors) if kaokore_searcher and kaokore_searcher.vectors is not None else 0

        return {
            "status": "healthy",
            "components": {
                "face_detector": detector_status,
                "face_encoder": encoder_status,
                "kaokore_searcher": kaokore_status,
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
        except Exception:
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
        if encoding is None or encoding.size == 0:
            raise HTTPException(status_code=400, detail="Failed to extract face features")

        # KaoKore類似検索
        kaokore_searcher = get_kaokore_similarity_searcher()
        similar_faces = kaokore_searcher.search_similar_faces(encoding, k=k)

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
async def get_metadata(image_id: str):
    """
    特定画像のメタデータを取得（KaoKoreデータセットから）

    Args:
        image_id: 画像ID

    Returns:
        画像のメタデータ
    """
    try:
        # KaoKoreローダーからメタデータを取得
        metadata = kaokore_loader.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Image metadata not found")

        return {"image_id": image_id, "metadata": metadata}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata for {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metadata")


@app.get("/api/v1/statistics")
async def get_statistics():
    """
    システム統計情報を取得（KaoKore専用）

    Returns:
        統計情報
    """
    try:
        # KaoKore類似検索の統計を取得
        kaokore_searcher = get_kaokore_similarity_searcher()

        # KaoKoreデータセットの統計
        kaokore_stats = kaokore_loader.get_statistics()
        available_images = len(kaokore_loader.get_available_images())

        # KaoKore処理済み画像数を取得
        processed_vectors = len(kaokore_searcher.vectors) if kaokore_searcher.vectors is not None else 0

        return {
            "search_engine": {
                "total_vectors": processed_vectors,
                "metadata_entries": kaokore_stats["total_images"],
                "vector_dimension": settings.face_vector_dimension,
            },
            "collection": {
                "downloaded_images": available_images,
                "total_metadata": kaokore_stats["total_images"],
            },
            "system": {
                "version": settings.app_version,
                "face_vector_dimension": settings.face_vector_dimension,
                "similarity_threshold": settings.similarity_threshold,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.get("/api/v1/kaokore/statistics")
async def get_kaokore_statistics():
    """
    KaoKoreデータセットの統計情報を取得

    Returns:
        KaoKoreデータセットの統計情報
    """
    try:
        stats = kaokore_loader.get_statistics()
        unique_tags = kaokore_loader.get_unique_tags()
        unique_sources = kaokore_loader.get_unique_sources()

        return {
            "dataset_name": "KaoKore Dataset",
            "statistics": stats,
            "unique_tags": unique_tags[:20],  # 最初の20個のタグ
            "total_unique_tags": len(unique_tags),
            "unique_sources": unique_sources,
            "available_images": len(kaokore_loader.get_available_images()),
        }

    except Exception as e:
        logger.error(f"Failed to get KaoKore statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve KaoKore statistics")


@app.get("/api/v1/kaokore/search/tag/{tag}")
async def search_kaokore_by_tag(tag: str):
    """
    タグでKaoKore画像を検索

    Args:
        tag: 検索するタグ

    Returns:
        タグに一致する画像一覧
    """
    try:
        results = kaokore_loader.search_by_tag(tag)

        return {
            "tag": tag,
            "total_results": len(results),
            "results": [
                {"filename": filename, "metadata": metadata}
                for filename, metadata in results[:50]  # 最初の50件
            ],
        }

    except Exception as e:
        logger.error(f"Failed to search KaoKore by tag {tag}: {e}")
        raise HTTPException(status_code=500, detail="Failed to search by tag")


@app.get("/api/v1/kaokore/image/{filename}")
async def get_kaokore_image(filename: str):
    """
    KaoKore画像ファイルを取得

    Args:
        filename: 画像ファイル名

    Returns:
        画像ファイル
    """
    try:
        image_path = kaokore_loader.get_image_path(filename)
        if not image_path or not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(path=str(image_path), media_type="image/jpeg", filename=filename)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get KaoKore image {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")


@app.get("/api/v1/kaokore/info/{filename}")
async def get_kaokore_image_info(filename: str):
    """
    KaoKore画像の詳細情報を取得

    Args:
        filename: 画像ファイル名

    Returns:
        画像の詳細情報
    """
    try:
        info = kaokore_loader.get_image_info(filename)
        if not info:
            raise HTTPException(status_code=404, detail="Image info not found")

        return {"filename": filename, "metadata": info}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get KaoKore image info {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image info")


if __name__ == "__main__":
    # 開発用サーバー起動
    uvicorn.run("src.api:app", host=settings.api_host, port=settings.api_port, reload=settings.debug, log_level="info")
