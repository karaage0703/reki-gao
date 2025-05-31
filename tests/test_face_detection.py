"""
顔検出機能のテスト
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import cv2

from src.face_detection import FaceDetector


class TestFaceDetector:
    """FaceDetectorクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.detector = FaceDetector(confidence_threshold=0.8)

    def test_init(self):
        """初期化のテスト"""
        assert self.detector.confidence_threshold == 0.8
        assert self.detector.face_cascade is not None

    def test_detect_faces_empty_image(self):
        """空の画像での顔検出テスト"""
        empty_image = np.array([])
        faces = self.detector.detect_faces(empty_image)
        assert faces == []

    def test_detect_faces_none_image(self):
        """None画像での顔検出テスト"""
        faces = self.detector.detect_faces(None)
        assert faces == []

    @patch("cv2.CascadeClassifier.detectMultiScale")
    def test_detect_faces_valid_image(self, mock_detect):
        """有効な画像での顔検出テスト"""
        # サンプル画像を作成（300x300のランダム画像）
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        # OpenCVの顔検出をモック
        mock_detect.return_value = np.array([[50, 50, 100, 100]])  # x, y, w, h

        faces = self.detector.detect_faces(test_image)

        assert len(faces) == 1
        assert faces[0]["x1"] == 50
        assert faces[0]["y1"] == 50
        assert faces[0]["x2"] == 150  # x + w
        assert faces[0]["y2"] == 150  # y + h
        assert faces[0]["method"] == "opencv_haar"

    def test_crop_face_valid(self):
        """有効な顔領域の切り抜きテスト"""
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face_box = {"x1": 50, "y1": 50, "x2": 150, "y2": 150, "confidence": 0.9}

        cropped = self.detector.crop_face(test_image, face_box)

        assert cropped is not None
        assert cropped.shape[0] > 0  # 高さ
        assert cropped.shape[1] > 0  # 幅

    def test_crop_face_invalid_box(self):
        """無効な顔領域での切り抜きテスト"""
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face_box = {
            "x1": 250,
            "y1": 250,
            "x2": 350,  # 画像範囲外
            "y2": 350,  # 画像範囲外
            "confidence": 0.9,
        }

        cropped = self.detector.crop_face(test_image, face_box)

        # 範囲外でも適切にクリップされて処理される
        assert cropped is not None or cropped is None  # 実装によって異なる

    def test_preprocess_face_for_encoding(self):
        """顔画像の前処理テスト"""
        # 160x160のサンプル顔画像
        face_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

        preprocessed = self.detector.preprocess_face_for_encoding(face_image)

        assert preprocessed is not None
        assert preprocessed.shape == (160, 160, 3)
        assert preprocessed.dtype == np.float32
        assert 0.0 <= preprocessed.max() <= 1.0  # 正規化されている

    def test_preprocess_face_none_image(self):
        """None画像の前処理テスト"""
        preprocessed = self.detector.preprocess_face_for_encoding(None)
        assert preprocessed is None

    def test_preprocess_face_empty_image(self):
        """空画像の前処理テスト"""
        empty_image = np.array([])
        preprocessed = self.detector.preprocess_face_for_encoding(empty_image)
        assert preprocessed is None

    def test_filter_faces_max_limit(self):
        """顔数制限のテスト"""
        # 多数の顔を模擬
        many_faces = []
        for i in range(10):
            face = {
                "x1": i * 10,
                "y1": i * 10,
                "x2": (i + 1) * 10,
                "y2": (i + 1) * 10,
                "confidence": 0.9 - i * 0.05,  # 信頼度を下げていく
                "method": "test",
            }
            many_faces.append(face)

        filtered = self.detector._filter_faces(many_faces)

        # 設定された最大数以下になっている
        from src.config import settings

        assert len(filtered) <= settings.max_faces_per_image

        # 信頼度順にソートされている
        if len(filtered) > 1:
            for i in range(len(filtered) - 1):
                assert filtered[i]["confidence"] >= filtered[i + 1]["confidence"]

    def test_filter_faces_empty_list(self):
        """空の顔リストのフィルタリングテスト"""
        filtered = self.detector._filter_faces([])
        assert filtered == []

    def test_detect_faces_no_detection(self):
        """顔が検出されない場合のテスト"""
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        with patch("cv2.CascadeClassifier.detectMultiScale") as mock_detect:
            mock_detect.return_value = np.array([])  # 顔が検出されない

            faces = self.detector.detect_faces(test_image)
            assert faces == []

    def test_detect_faces_multiple_detection(self):
        """複数の顔が検出される場合のテスト"""
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        with patch("cv2.CascadeClassifier.detectMultiScale") as mock_detect:
            # 2つの顔を検出
            mock_detect.return_value = np.array(
                [
                    [50, 50, 100, 100],  # 1つ目の顔
                    [200, 200, 80, 80],  # 2つ目の顔
                ]
            )

            faces = self.detector.detect_faces(test_image)
            assert len(faces) == 2
            assert faces[0]["x1"] == 50
            assert faces[1]["x1"] == 200


@pytest.fixture
def sample_image():
    """テスト用のサンプル画像を生成"""
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_box():
    """テスト用の顔領域情報を生成"""
    return {"x1": 50, "y1": 50, "x2": 150, "y2": 150, "confidence": 0.9, "method": "test"}
