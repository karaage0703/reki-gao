"""
KaoKoreデータセット用類似顔検索機能
"""

import numpy as np
import cv2
from typing import List, Dict, Optional
from pathlib import Path
import logging
from sklearn.neighbors import NearestNeighbors

from .config import settings
from .kaokore_loader import kaokore_loader
from .face_detection import FaceDetector
from .face_encoding import FaceEncoder

logger = logging.getLogger(__name__)


class KaoKoreSimilaritySearcher:
    """KaoKoreデータセット用類似顔検索クラス"""

    def __init__(self, max_images: Optional[int] = None):
        """
        初期化

        Args:
            max_images: 処理する最大画像数（テスト用、Noneで全画像）
        """
        self.face_detector = FaceDetector()
        self.face_encoder = FaceEncoder()
        self.model = None
        self.vectors = None
        self.image_files = []
        self.metadata_list = []
        self.max_images = max_images

        # KaoKoreデータセットのベクトルを構築
        self._build_kaokore_index()

    def _build_kaokore_index(self):
        """KaoKoreデータセットのインデックスを構築"""
        try:
            logger.info("Building KaoKore similarity index...")

            # 利用可能な画像を取得
            available_images = kaokore_loader.get_available_images()

            # テスト用に画像数を制限
            if self.max_images is not None:
                available_images = available_images[: self.max_images]
                logger.info(f"Processing {len(available_images)} KaoKore images (limited for testing)...")
            else:
                logger.info(f"Processing {len(available_images)} KaoKore images...")

            vectors = []
            valid_images = []
            valid_metadata = []

            for i, image_file in enumerate(available_images):
                if i % 100 == 0:
                    logger.info(f"Processing image {i + 1}/{len(available_images)}")

                try:
                    # 画像パス
                    image_path = Path(settings.kaokore_images_dir) / image_file

                    if not image_path.exists():
                        continue

                    # 画像を読み込み
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue

                    # 画像を前処理（KaoKoreは既に顔画像なので顔検出不要）
                    preprocessed_face = self.face_detector.preprocess_face_for_encoding(image)
                    if preprocessed_face is None:
                        continue

                    # 顔エンコーディング
                    face_vector = self.face_encoder.encode_face(preprocessed_face)
                    if face_vector is None or face_vector.size == 0:
                        continue

                    # メタデータを取得
                    metadata = kaokore_loader.get_image_metadata(image_file)

                    vectors.append(face_vector)
                    valid_images.append(image_file)
                    valid_metadata.append(metadata)

                except Exception as e:
                    logger.warning(f"Failed to process {image_file}: {e}")
                    continue

            if not vectors:
                logger.error("No valid face vectors found in KaoKore dataset")
                return

            # ベクトルを正規化
            self.vectors = np.array(vectors).astype(np.float32)
            for i in range(len(self.vectors)):
                norm = np.linalg.norm(self.vectors[i])
                if norm > 0:
                    self.vectors[i] = self.vectors[i] / norm

            self.image_files = valid_images
            self.metadata_list = valid_metadata

            # NearestNeighborsモデルを構築
            k = min(settings.similarity_search_k, len(self.vectors))
            self.model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            self.model.fit(self.vectors)

            logger.info(f"KaoKore similarity index built with {len(self.vectors)} face vectors")

        except Exception as e:
            logger.error(f"Failed to build KaoKore index: {e}")

    def search_similar_faces(self, query_vector: np.ndarray, k: int = None) -> List[Dict]:
        """
        類似顔を検索する

        Args:
            query_vector: クエリ特徴量ベクトル
            k: 返す結果数

        Returns:
            類似顔のリスト
        """
        if k is None:
            k = settings.similarity_search_k

        if self.model is None or self.vectors is None:
            logger.warning("KaoKore index not available")
            return []

        try:
            # ベクトルの正規化
            query_vector = query_vector.astype(np.float32)
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

            # 検索実行
            query_vector = query_vector.reshape(1, -1)
            k = min(k, len(self.vectors))
            distances, indices = self.model.kneighbors(query_vector, n_neighbors=k)

            # コサイン類似度を計算
            similarities = 1 - distances[0]

            # 結果を整理
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities, indices[0])):
                # 類似度が閾値以下の場合はスキップ
                if similarity < settings.similarity_threshold:
                    continue

                metadata = self.metadata_list[idx]
                image_file = self.image_files[idx]

                # 日本語メタデータを英語表示用にマッピング
                gender_map = {"男": "男性", "女": "女性"}
                status_map = {"貴族": "貴族", "武士": "武士", "庶民": "庶民"}

                # 原典から時代を推定
                def estimate_era_from_source(source: str) -> str:
                    """原典名から時代を推定"""
                    if not source or source == "":
                        return "不明"

                    # 平安時代の作品
                    heian_works = ["源氏物語", "伊勢物語", "竹取物語", "宇津保物語", "狭衣"]
                    # 鎌倉時代の作品
                    kamakura_works = ["平家物語", "方丈記", "徒然草", "後三年合戦絵", "前九年軍記"]
                    # 室町時代の作品
                    muromachi_works = ["酒呑童子", "土くも", "羅生門"]
                    # 江戸時代の作品
                    edo_works = ["雛屋立圃絵入書巻"]

                    for work in heian_works:
                        if work in source:
                            return "平安時代"
                    for work in kamakura_works:
                        if work in source:
                            return "鎌倉時代"
                    for work in muromachi_works:
                        if work in source:
                            return "室町時代"
                    for work in edo_works:
                        if work in source:
                            return "江戸時代"

                    # 絵巻物は主に平安〜鎌倉時代
                    if "絵巻" in source:
                        return "平安〜鎌倉時代"
                    # 草子は主に室町時代
                    if "草子" in source or "さうし" in source:
                        return "室町時代"
                    # 物語は主に平安時代
                    if "物語" in source:
                        return "平安時代"

                    return "中世"

                era = metadata.get("制作年", "")
                if not era or era.strip() == "":
                    era = estimate_era_from_source(metadata.get("原典", ""))

                result = {
                    "rank": i + 1,
                    "similarity": float(similarity),
                    "index": int(idx),
                    "image_id": image_file.replace(".jpg", ""),
                    "image_url": f"/api/v1/kaokore/image/{image_file}",
                    "person_name": metadata.get("タグ", "不明"),  # タグを人物名として使用
                    "tags": metadata.get("タグ", ""),  # タグ情報を別途追加
                    "era": era,
                    "source": metadata.get("原典", "KaoKore Dataset"),
                    "collection": metadata.get("所蔵", "ROIS-CODH"),
                    "license": "CC BY-SA 4.0",
                    "gender": gender_map.get(metadata.get("性別", ""), metadata.get("性別", "不明")),
                    "status": status_map.get(metadata.get("身分", ""), metadata.get("身分", "不明")),
                    "metadata": metadata,
                }

                results.append(result)

            logger.info(f"Found {len(results)} similar faces in KaoKore dataset")
            return results

        except Exception as e:
            logger.error(f"KaoKore similarity search failed: {e}")
            return []

    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return {
            "total_vectors": len(self.vectors) if self.vectors is not None else 0,
            "total_images": len(self.image_files),
            "vector_dimension": settings.face_vector_dimension,
            "dataset": "KaoKore",
        }


# グローバルインスタンス
kaokore_similarity_searcher = None


def get_kaokore_similarity_searcher(max_images: Optional[int] = None, force_reinit: bool = False) -> KaoKoreSimilaritySearcher:
    """
    KaoKore類似検索インスタンスを取得

    Args:
        max_images: 処理する最大画像数（テスト用、Noneで全画像）
        force_reinit: 強制的に再初期化するかどうか
    """
    global kaokore_similarity_searcher
    if kaokore_similarity_searcher is None or force_reinit:
        from .config import settings

        # 引数 > 設定ファイル > デフォルト(100)の優先順位
        if max_images is not None:
            test_max_images = max_images
        else:
            test_max_images = settings.kaokore_max_images

        logger.info(f"Initializing KaoKore searcher with max_images: {test_max_images}")
        kaokore_similarity_searcher = KaoKoreSimilaritySearcher(max_images=test_max_images)
    return kaokore_similarity_searcher


def reset_kaokore_similarity_searcher():
    """KaoKore類似検索インスタンスをリセット"""
    global kaokore_similarity_searcher
    kaokore_similarity_searcher = None
