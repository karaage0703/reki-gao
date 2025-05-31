"""
FastAPI エンドポイントのテスト
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import io
from PIL import Image

from src.api import app


class TestAPI:
    """APIエンドポイントのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """ルートエンドポイントのテスト"""
        response = self.client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "reki-gao API" in data["message"]
        assert "version" in data
        assert "docs" in data

    @patch("src.api.face_detector")
    @patch("src.api.face_encoder")
    @patch("src.api.similarity_searcher")
    @patch("src.api.ganbo_manager")
    def test_health_check_healthy(self, mock_manager, mock_searcher, mock_encoder, mock_detector):
        """ヘルスチェック（正常）のテスト"""
        # モックの設定
        mock_detector.__bool__ = Mock(return_value=True)
        mock_encoder.__bool__ = Mock(return_value=True)
        mock_searcher.__bool__ = Mock(return_value=True)
        mock_manager.__bool__ = Mock(return_value=True)
        mock_searcher.get_index_size.return_value = 100

        response = self.client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["face_detector"] is True
        assert data["components"]["face_encoder"] is True
        assert data["components"]["similarity_searcher"] is True
        assert data["components"]["ganbo_manager"] is True
        assert data["index_size"] == 100

    def test_health_check_unhealthy(self):
        """ヘルスチェック（異常）のテスト"""
        # グローバル変数をNoneに設定
        with patch("src.api.face_detector", None):
            with patch("src.api.face_encoder", None):
                with patch("src.api.similarity_searcher", None):
                    with patch("src.api.ganbo_manager", None):
                        response = self.client.get("/api/v1/health")
                        assert response.status_code == 200

                        data = response.json()
                        assert data["status"] == "healthy"
                        assert data["components"]["face_detector"] is False
                        assert data["components"]["face_encoder"] is False
                        assert data["components"]["similarity_searcher"] is False
                        assert data["components"]["ganbo_manager"] is False
                        assert data["index_size"] == 0

    def create_test_image(self, width=300, height=300):
        """テスト用画像を作成"""
        # PIL画像を作成
        image = Image.new("RGB", (width, height), color="red")

        # バイトストリームに変換
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        return img_byte_arr

    @patch("src.api.get_face_detector")
    @patch("src.api.get_face_encoder")
    @patch("src.api.get_similarity_searcher")
    def test_upload_and_search_success(self, mock_get_searcher, mock_get_encoder, mock_get_detector):
        """画像アップロード・検索成功のテスト"""
        # モックの設定
        mock_detector = Mock()
        mock_encoder = Mock()
        mock_searcher = Mock()

        mock_get_detector.return_value = mock_detector
        mock_get_encoder.return_value = mock_encoder
        mock_get_searcher.return_value = mock_searcher

        # 顔検出の結果
        mock_faces = [{"x1": 50, "y1": 50, "x2": 150, "y2": 150, "confidence": 0.9, "method": "test"}]
        mock_detector.detect_faces.return_value = mock_faces

        # 顔切り抜きの結果
        mock_face_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_detector.crop_face.return_value = mock_face_crop

        # 前処理の結果
        mock_preprocessed = np.random.rand(160, 160, 3).astype(np.float32)
        mock_detector.preprocess_face_for_encoding.return_value = mock_preprocessed

        # 特徴量抽出の結果
        mock_encoding = np.random.rand(128).astype(np.float32)
        mock_encoder.encode_face.return_value = mock_encoding
        mock_encoder.is_valid_encoding.return_value = True

        # 類似検索の結果
        mock_similar_faces = [
            {"rank": 1, "similarity": 0.85, "person_name": "織田信長", "era": "戦国時代", "image_id": "ganbo_0001"}
        ]
        mock_searcher.search_similar_faces.return_value = mock_similar_faces

        # テスト画像を作成
        test_image = self.create_test_image()

        # APIリクエスト
        response = self.client.post("/api/v1/upload", files={"file": ("test.jpg", test_image, "image/jpeg")}, params={"k": 3})

        assert response.status_code == 200

        data = response.json()
        assert "detected_faces" in data
        assert "main_face" in data
        assert "similar_faces" in data
        assert data["detected_faces"] == 1
        assert len(data["similar_faces"]) == 1
        assert data["similar_faces"][0]["person_name"] == "織田信長"

    def test_upload_invalid_file_type(self):
        """無効なファイル形式のテスト"""
        # テキストファイルを送信
        text_content = io.BytesIO(b"This is not an image")

        response = self.client.post("/api/v1/upload", files={"file": ("test.txt", text_content, "text/plain")})

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @patch("src.api.get_face_detector")
    def test_upload_no_face_detected(self, mock_get_detector):
        """顔が検出されない場合のテスト"""
        mock_detector = Mock()
        mock_get_detector.return_value = mock_detector

        # 顔が検出されない
        mock_detector.detect_faces.return_value = []

        test_image = self.create_test_image()

        response = self.client.post("/api/v1/upload", files={"file": ("test.jpg", test_image, "image/jpeg")})

        assert response.status_code == 400
        assert "No face detected" in response.json()["detail"]

    def test_upload_file_too_large(self):
        """ファイルサイズが大きすぎる場合のテスト"""
        # 大きな画像を作成（設定値を超える）
        large_image = self.create_test_image(width=5000, height=5000)

        # ファイルサイズ制限をモック
        with patch("src.api.settings.max_file_size", 1000):  # 1KB制限
            response = self.client.post("/api/v1/upload", files={"file": ("large.jpg", large_image, "image/jpeg")})

            assert response.status_code == 400
            assert "File size too large" in response.json()["detail"]

    @patch("src.api.get_ganbo_manager")
    def test_get_metadata_success(self, mock_get_manager):
        """メタデータ取得成功のテスト"""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        # メタデータの結果
        mock_metadata = {"image_id": "ganbo_0001", "person_name": "織田信長", "era": "戦国時代", "source": "本能寺の変図屏風"}
        mock_manager.get_metadata.return_value = mock_metadata

        response = self.client.get("/api/v1/metadata/ganbo_0001")

        assert response.status_code == 200
        data = response.json()
        assert data["image_id"] == "ganbo_0001"
        assert data["person_name"] == "織田信長"

    @patch("src.api.get_ganbo_manager")
    def test_get_metadata_not_found(self, mock_get_manager):
        """メタデータが見つからない場合のテスト"""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        # メタデータが見つからない
        mock_manager.get_metadata.return_value = {}

        response = self.client.get("/api/v1/metadata/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("src.api.get_similarity_searcher")
    @patch("src.api.get_ganbo_manager")
    def test_get_statistics(self, mock_get_manager, mock_get_searcher):
        """統計情報取得のテスト"""
        mock_searcher = Mock()
        mock_manager = Mock()
        mock_get_searcher.return_value = mock_searcher
        mock_get_manager.return_value = mock_manager

        # 統計情報の結果
        mock_search_stats = {"total_vectors": 100, "vector_dimension": 128}
        mock_collection_stats = {"total_images": 100, "downloaded_images": 95}

        mock_searcher.get_statistics.return_value = mock_search_stats
        mock_manager.get_statistics.return_value = mock_collection_stats

        response = self.client.get("/api/v1/statistics")

        assert response.status_code == 200
        data = response.json()
        assert "search_engine" in data
        assert "collection" in data
        assert "system" in data
        assert data["search_engine"]["total_vectors"] == 100
        assert data["collection"]["total_images"] == 100

    @patch("src.api.build_initial_index")
    @patch("src.api.get_similarity_searcher")
    def test_rebuild_index(self, mock_get_searcher, mock_build_index):
        """インデックス再構築のテスト"""
        mock_searcher = Mock()
        mock_get_searcher.return_value = mock_searcher
        mock_searcher.get_index_size.return_value = 150

        # 非同期関数のモック
        mock_build_index.return_value = AsyncMock()

        response = self.client.post("/api/v1/admin/rebuild-index")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "rebuilt successfully" in data["message"]
        assert data["new_index_size"] == 150


@pytest.fixture
def test_client():
    """テスト用のFastAPIクライアント"""
    return TestClient(app)


@pytest.fixture
def sample_image_file():
    """テスト用の画像ファイル"""
    image = Image.new("RGB", (300, 300), color="blue")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.fixture
def mock_face_detection_result():
    """テスト用の顔検出結果"""
    return [{"x1": 50, "y1": 50, "x2": 150, "y2": 150, "confidence": 0.9, "method": "test"}]


@pytest.fixture
def mock_similarity_search_result():
    """テスト用の類似検索結果"""
    return [
        {"rank": 1, "similarity": 0.85, "person_name": "織田信長", "era": "戦国時代", "image_id": "ganbo_0001"},
        {"rank": 2, "similarity": 0.78, "person_name": "豊臣秀吉", "era": "安土桃山時代", "image_id": "ganbo_0002"},
    ]
