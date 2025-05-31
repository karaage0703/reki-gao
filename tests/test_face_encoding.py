"""
顔特徴量抽出機能のテスト
"""

import pytest
import numpy as np
from unittest.mock import patch

from src.face_encoding import FaceEncoder, FaceEncodingCache


class TestFaceEncoder:
    """FaceEncoderクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.encoder = FaceEncoder(model_name="face_recognition")

    def test_init_face_recognition(self):
        """face_recognitionモデルでの初期化テスト"""
        encoder = FaceEncoder(model_name="face_recognition")
        assert encoder.model_name == "face_recognition"
        assert encoder.model == "face_recognition"

    def test_init_facenet(self):
        """FaceNetモデルでの初期化テスト"""
        encoder = FaceEncoder(model_name="Facenet")
        assert encoder.model_name == "Facenet"
        assert encoder.model == "facenet"

    def test_init_default(self):
        """デフォルトモデルでの初期化テスト"""
        encoder = FaceEncoder(model_name="unknown")
        assert encoder.model == "facenet"  # デフォルトはfacenet

    def test_encode_face_none_image(self):
        """None画像での特徴量抽出テスト"""
        encoding = self.encoder.encode_face(None)
        assert encoding is None

    def test_encode_face_empty_image(self):
        """空画像での特徴量抽出テスト"""
        empty_image = np.array([])
        encoding = self.encoder.encode_face(empty_image)
        assert encoding is None

    @patch("face_recognition.face_locations")
    @patch("face_recognition.face_encodings")
    def test_encode_with_face_recognition_success(self, mock_encodings, mock_locations):
        """face_recognitionでの成功ケーステスト"""
        # モックの設定
        mock_locations.return_value = [(0, 160, 160, 0)]  # 1つの顔
        mock_encoding = np.random.rand(128).astype(np.float32)
        mock_encodings.return_value = [mock_encoding]

        # テスト画像
        test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

        encoding = self.encoder.encode_face(test_image)

        assert encoding is not None
        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float32
        assert len(encoding) == 128

        # 正規化されているかチェック
        assert abs(np.linalg.norm(encoding) - 1.0) < 1e-6

    @patch("face_recognition.face_locations")
    def test_encode_with_face_recognition_no_face(self, mock_locations):
        """face_recognitionで顔が検出されない場合のテスト"""
        mock_locations.return_value = []  # 顔なし

        test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

        encoding = self.encoder.encode_face(test_image)
        assert encoding is None

    @patch("face_recognition.face_encodings")
    @patch("face_recognition.face_locations")
    def test_encode_with_face_recognition_no_encoding(self, mock_locations, mock_encodings):
        """face_recognitionで特徴量抽出に失敗する場合のテスト"""
        mock_locations.return_value = [(0, 160, 160, 0)]
        mock_encodings.return_value = []  # 特徴量抽出失敗

        test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

        encoding = self.encoder.encode_face(test_image)
        assert encoding is None

    def test_encode_faces_batch(self):
        """バッチ処理のテスト"""
        # テスト画像のリスト
        test_images = [
            np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8),
            np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8),
            None,  # 無効な画像
            np.array([]),  # 空の画像
        ]

        with patch.object(self.encoder, "encode_face") as mock_encode:
            # モックの戻り値を設定
            mock_encode.side_effect = [
                np.random.rand(128).astype(np.float32),  # 成功
                np.random.rand(128).astype(np.float32),  # 成功
                None,  # 失敗
                None,  # 失敗
            ]

            encodings = self.encoder.encode_faces_batch(test_images)

            assert len(encodings) == 4
            assert encodings[0] is not None
            assert encodings[1] is not None
            assert encodings[2] is None
            assert encodings[3] is None

    def test_calculate_similarity_cosine(self):
        """コサイン類似度計算のテスト"""
        # 同じベクトル
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        similarity = self.encoder.calculate_similarity(vec1, vec2, method="cosine")
        assert abs(similarity - 1.0) < 1e-6

        # 直交ベクトル
        vec3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = self.encoder.calculate_similarity(vec1, vec3, method="cosine")
        assert abs(similarity - 0.0) < 1e-6

    def test_calculate_similarity_euclidean(self):
        """ユークリッド距離による類似度計算のテスト"""
        # 同じベクトル
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        similarity = self.encoder.calculate_similarity(vec1, vec2, method="euclidean")
        assert similarity == 0.5  # 距離0の場合の類似度

        # 異なるベクトル
        vec3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        similarity = self.encoder.calculate_similarity(vec1, vec3, method="euclidean")
        assert 0.0 < similarity < 1.0

    def test_calculate_similarity_none_vectors(self):
        """Noneベクトルでの類似度計算テスト"""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        similarity = self.encoder.calculate_similarity(None, vec1)
        assert similarity == 0.0

        similarity = self.encoder.calculate_similarity(vec1, None)
        assert similarity == 0.0

        similarity = self.encoder.calculate_similarity(None, None)
        assert similarity == 0.0

    def test_calculate_similarity_unknown_method(self):
        """未知の類似度計算方法のテスト"""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        similarity = self.encoder.calculate_similarity(vec1, vec2, method="unknown")
        assert similarity == 0.0

    def test_is_valid_encoding(self):
        """特徴量ベクトルの有効性チェックテスト"""
        # 有効なベクトル
        valid_encoding = np.random.rand(128).astype(np.float32)
        assert self.encoder.is_valid_encoding(valid_encoding) is True

        # None
        assert self.encoder.is_valid_encoding(None) is False

        # 空配列
        empty_array = np.array([])
        assert self.encoder.is_valid_encoding(empty_array) is False

        # NaNを含む
        nan_array = np.array([1.0, np.nan, 3.0])
        assert self.encoder.is_valid_encoding(nan_array) is False

        # 無限大を含む
        inf_array = np.array([1.0, np.inf, 3.0])
        assert self.encoder.is_valid_encoding(inf_array) is False

        # ゼロベクトル
        zero_array = np.zeros(128)
        assert self.encoder.is_valid_encoding(zero_array) is False

        # 非numpy配列
        assert self.encoder.is_valid_encoding([1, 2, 3]) is False


class TestFaceEncodingCache:
    """FaceEncodingCacheクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.cache = FaceEncodingCache()

    def test_cache_operations(self):
        """キャッシュの基本操作テスト"""
        # 初期状態
        assert self.cache.size() == 0
        assert self.cache.get("test_hash") is None

        # データ追加
        test_encoding = np.random.rand(128).astype(np.float32)
        self.cache.set("test_hash", test_encoding)

        assert self.cache.size() == 1
        retrieved = self.cache.get("test_hash")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, test_encoding)

        # クリア
        self.cache.clear()
        assert self.cache.size() == 0
        assert self.cache.get("test_hash") is None

    def test_cache_multiple_items(self):
        """複数アイテムのキャッシュテスト"""
        encodings = {}
        for i in range(5):
            hash_key = f"hash_{i}"
            encoding = np.random.rand(128).astype(np.float32)
            encodings[hash_key] = encoding
            self.cache.set(hash_key, encoding)

        assert self.cache.size() == 5

        # 全て取得できることを確認
        for hash_key, expected_encoding in encodings.items():
            retrieved = self.cache.get(hash_key)
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, expected_encoding)


@pytest.fixture
def sample_face_image():
    """テスト用のサンプル顔画像を生成"""
    return np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)


@pytest.fixture
def sample_encoding():
    """テスト用のサンプル特徴量ベクトルを生成"""
    encoding = np.random.rand(128).astype(np.float32)
    return encoding / np.linalg.norm(encoding)  # 正規化
