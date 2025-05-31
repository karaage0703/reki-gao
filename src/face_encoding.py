"""
顔特徴量抽出機能
OpenCVを使用した簡単な顔の特徴量ベクトル抽出
"""

import numpy as np
import cv2
from typing import Optional, List
import logging

from .config import settings

logger = logging.getLogger(__name__)


class FaceEncoder:
    """顔の特徴量抽出を行うクラス"""

    def __init__(self, model_name: str = "opencv_lbp"):
        """
        初期化メソッド

        Args:
            model_name: 使用するモデル名（opencv_lbp）
        """
        self.model_name = model_name
        self.vector_dimension = settings.face_vector_dimension

        logger.info(f"FaceEncoder initialized with model: {model_name}")

    def encode_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        顔画像から特徴量ベクトルを抽出

        Args:
            face_image: 前処理済み顔画像

        Returns:
            特徴量ベクトル（128次元）
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Invalid face image for encoding")
            return None

        try:
            return self._encode_with_opencv_lbp(face_image)

        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return None

    def _encode_with_opencv_lbp(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """OpenCVのLBP（Local Binary Pattern）を使用した特徴量抽出"""
        try:
            # グレースケール変換
            if len(face_image.shape) == 3:
                if face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                if face_image.max() <= 1.0:
                    gray = (face_image * 255).astype(np.uint8)
                else:
                    gray = face_image.astype(np.uint8)

            # 画像を固定サイズにリサイズ
            target_size = (64, 64)
            resized = cv2.resize(gray, target_size)

            # ヒストグラム均等化
            equalized = cv2.equalizeHist(resized)

            # LBP（Local Binary Pattern）特徴量を計算
            lbp_features = self._calculate_lbp_features(equalized)

            # HOG（Histogram of Oriented Gradients）特徴量を計算
            hog_features = self._calculate_hog_features(equalized)

            # 統計的特徴量を計算
            stat_features = self._calculate_statistical_features(equalized)

            # 特徴量を結合
            combined_features = np.concatenate([lbp_features, hog_features, stat_features])

            # 指定された次元数に調整
            if len(combined_features) > self.vector_dimension:
                # PCAの代わりに単純に切り詰め
                vector = combined_features[: self.vector_dimension]
            else:
                # パディング
                vector = np.pad(combined_features, (0, self.vector_dimension - len(combined_features)), "constant")

            # 正規化
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)

            return vector.astype(np.float32)

        except Exception as e:
            logger.error(f"OpenCV LBP encoding failed: {e}")
            return None

    def _calculate_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """LBP特徴量を計算"""
        try:
            # 簡単なLBP実装
            h, w = image.shape
            lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i - 1, j - 1] >= center) << 7
                    code |= (image[i - 1, j] >= center) << 6
                    code |= (image[i - 1, j + 1] >= center) << 5
                    code |= (image[i, j + 1] >= center) << 4
                    code |= (image[i + 1, j + 1] >= center) << 3
                    code |= (image[i + 1, j] >= center) << 2
                    code |= (image[i + 1, j - 1] >= center) << 1
                    code |= (image[i, j - 1] >= center) << 0
                    lbp[i - 1, j - 1] = code

            # ヒストグラムを計算
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            return hist.astype(np.float32)

        except Exception as e:
            logger.error(f"LBP calculation failed: {e}")
            return np.zeros(256, dtype=np.float32)

    def _calculate_hog_features(self, image: np.ndarray) -> np.ndarray:
        """HOG特徴量を計算"""
        try:
            # HOGディスクリプタを作成
            hog = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

            # HOG特徴量を計算
            features = hog.compute(image)
            return features.flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"HOG calculation failed: {e}")
            return np.zeros(324, dtype=np.float32)  # デフォルトサイズ

    def _calculate_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """統計的特徴量を計算"""
        try:
            features = []

            # 基本統計量
            features.append(np.mean(image))
            features.append(np.std(image))
            features.append(np.min(image))
            features.append(np.max(image))
            features.append(np.median(image))

            # 画像の各象限の平均値
            h, w = image.shape
            h_half, w_half = h // 2, w // 2

            features.append(np.mean(image[:h_half, :w_half]))  # 左上
            features.append(np.mean(image[:h_half, w_half:]))  # 右上
            features.append(np.mean(image[h_half:, :w_half]))  # 左下
            features.append(np.mean(image[h_half:, w_half:]))  # 右下

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Statistical features calculation failed: {e}")
            return np.zeros(9, dtype=np.float32)

    def encode_faces_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        複数の顔画像を一括で特徴量抽出

        Args:
            face_images: 顔画像のリスト

        Returns:
            特徴量ベクトルのリスト
        """
        encodings = []

        for i, face_image in enumerate(face_images):
            logger.debug(f"Encoding face {i + 1}/{len(face_images)}")
            encoding = self.encode_face(face_image)
            encodings.append(encoding)

        logger.info(f"Encoded {len([e for e in encodings if e is not None])}/{len(face_images)} faces")
        return encodings

    def calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray, method: str = "cosine") -> float:
        """
        2つの特徴量ベクトル間の類似度を計算

        Args:
            encoding1: 特徴量ベクトル1
            encoding2: 特徴量ベクトル2
            method: 類似度計算方法（"cosine" or "euclidean"）

        Returns:
            類似度スコア
        """
        try:
            if encoding1 is None or encoding2 is None:
                return 0.0

            if method == "cosine":
                # コサイン類似度
                similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
                return float(similarity)

            elif method == "euclidean":
                # ユークリッド距離（類似度に変換）
                distance = np.linalg.norm(encoding1 - encoding2)
                # 距離を類似度に変換（0-1の範囲）
                similarity = 1.0 / (1.0 + distance)
                return float(similarity)

            else:
                logger.error(f"Unknown similarity method: {method}")
                return 0.0

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def is_valid_encoding(self, encoding: np.ndarray) -> bool:
        """
        特徴量ベクトルが有効かチェック

        Args:
            encoding: 特徴量ベクトル

        Returns:
            有効性（True/False）
        """
        if encoding is None:
            return False

        if not isinstance(encoding, np.ndarray):
            return False

        if encoding.size == 0:
            return False

        if np.isnan(encoding).any() or np.isinf(encoding).any():
            return False

        # ゼロベクトルチェック
        if np.allclose(encoding, 0):
            return False

        return True


class FaceEncodingCache:
    """顔特徴量のキャッシュクラス"""

    def __init__(self):
        self.cache = {}

    def get(self, image_hash: str) -> Optional[np.ndarray]:
        """キャッシュから特徴量を取得"""
        return self.cache.get(image_hash)

    def set(self, image_hash: str, encoding: np.ndarray):
        """キャッシュに特徴量を保存"""
        self.cache[image_hash] = encoding

    def clear(self):
        """キャッシュをクリア"""
        self.cache.clear()

    def size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self.cache)


# グローバルキャッシュインスタンス
encoding_cache = FaceEncodingCache()
