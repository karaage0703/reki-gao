"""
顔検出機能
OpenCVのHaar Cascadeを使用した顔検出・切り抜き処理
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .config import settings

logger = logging.getLogger(__name__)


class FaceDetector:
    """顔検出を行うクラス"""

    def __init__(self, confidence_threshold: float = None):
        """
        初期化メソッド

        Args:
            confidence_threshold: 顔検出の信頼度閾値
        """
        self.confidence_threshold = confidence_threshold or settings.face_confidence_threshold

        # OpenCV Haar Cascade顔検出器の初期化
        self.face_cascade = None
        self._init_opencv_detector()

        logger.info(f"FaceDetector initialized with confidence threshold: {self.confidence_threshold}")

    def _init_opencv_detector(self):
        """OpenCV Haar Cascade顔検出器を初期化"""
        try:
            # OpenCVの事前学習済みHaar Cascadeを使用
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            if self.face_cascade.empty():
                logger.error("Failed to load Haar Cascade classifier")
                self.face_cascade = None
            else:
                logger.info("OpenCV Haar Cascade face detector loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenCV detector: {e}")
            self.face_cascade = None

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        画像から顔を検出する

        Args:
            image: 入力画像（numpy array）

        Returns:
            検出された顔のリスト（座標、信頼度を含む）
        """
        if image is None or image.size == 0:
            logger.warning("Invalid input image")
            return []

        faces = []

        # OpenCV Haar Cascade検出器を使用
        if self.face_cascade is not None:
            opencv_faces = self._detect_with_opencv(image)
            faces.extend(opencv_faces)

        # 信頼度でフィルタリング
        faces = self._filter_faces(faces)

        logger.info(f"Detected {len(faces)} faces")
        return faces

    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict]:
        """OpenCV Haar Cascadeを使用した顔検出"""
        try:
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 顔検出
            faces_opencv = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
            )

            faces = []
            for x, y, w, h in faces_opencv:
                # 信頼度は固定値（Haar Cascadeは信頼度を返さない）
                confidence = 0.85

                faces.append(
                    {
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h,
                        "confidence": confidence,
                        "method": "opencv_haar",
                    }
                )

            return faces

        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []

    def _filter_faces(self, faces: List[Dict]) -> List[Dict]:
        """重複除去と信頼度フィルタリング"""
        if not faces:
            return []

        # 信頼度でフィルタリング
        filtered_faces = [face for face in faces if face["confidence"] >= self.confidence_threshold]

        # 信頼度でソート
        filtered_faces = sorted(filtered_faces, key=lambda x: x["confidence"], reverse=True)

        # 最大顔数制限
        filtered_faces = filtered_faces[: settings.max_faces_per_image]

        # TODO: IoU（Intersection over Union）による重複除去を実装

        return filtered_faces

    def crop_face(self, image: np.ndarray, face_box: Dict, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        顔領域を切り抜く

        Args:
            image: 元画像
            face_box: 顔の座標情報
            padding: 顔領域の拡張率

        Returns:
            切り抜かれた顔画像
        """
        try:
            h, w = image.shape[:2]

            # 座標取得
            x1, y1, x2, y2 = face_box["x1"], face_box["y1"], face_box["x2"], face_box["y2"]

            # パディング追加
            face_w = x2 - x1
            face_h = y2 - y1
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)

            # 拡張された座標
            x1_padded = max(0, x1 - pad_w)
            y1_padded = max(0, y1 - pad_h)
            x2_padded = min(w, x2 + pad_w)
            y2_padded = min(h, y2 + pad_h)

            # 顔領域を切り抜き
            face_crop = image[y1_padded:y2_padded, x1_padded:x2_padded]

            if face_crop.size == 0:
                logger.warning("Empty face crop")
                return None

            return face_crop

        except Exception as e:
            logger.error(f"Face cropping failed: {e}")
            return None

    def preprocess_face_for_encoding(
        self, face_image: np.ndarray, target_size: Tuple[int, int] = (160, 160)
    ) -> Optional[np.ndarray]:
        """
        顔画像を特徴量抽出用に前処理

        Args:
            face_image: 顔画像
            target_size: リサイズ後のサイズ

        Returns:
            前処理済み顔画像
        """
        try:
            if face_image is None or face_image.size == 0:
                return None

            # リサイズ
            resized = cv2.resize(face_image, target_size)

            # 正規化
            normalized = resized.astype(np.float32) / 255.0

            # RGB変換（OpenCVはBGRなので）
            if len(normalized.shape) == 3:
                normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)

            return normalized

        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None
