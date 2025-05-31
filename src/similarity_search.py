"""
類似顔検索機能
scikit-learnのNearestNeighborsを使用した高速ベクトル検索
"""

import numpy as np
import json
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings

logger = logging.getLogger(__name__)


class SimilaritySearcher:
    """類似顔検索を行うクラス"""

    def __init__(self, index_path: str = None, metadata_path: str = None):
        """
        初期化メソッド

        Args:
            index_path: インデックスファイルのパス
            metadata_path: メタデータファイルのパス
        """
        self.index_path = index_path or settings.faiss_index_path.replace(".faiss", ".pkl")
        self.metadata_path = metadata_path or settings.metadata_path

        self.model = None
        self.vectors = None
        self.metadata = {}
        self.vector_dimension = settings.face_vector_dimension

        # インデックスとメタデータの読み込み
        self._load_index()
        self._load_metadata()

        logger.info(f"SimilaritySearcher initialized with {self.get_index_size()} vectors")

    def _load_index(self):
        """インデックスを読み込み"""
        try:
            index_file = Path(self.index_path)
            if index_file.exists():
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                    self.vectors = data.get("vectors")
                    if self.vectors is not None and len(self.vectors) > 0:
                        self.model = NearestNeighbors(
                            n_neighbors=min(settings.similarity_search_k, len(self.vectors)),
                            metric="cosine",
                            algorithm="brute",
                        )
                        self.model.fit(self.vectors)
                        logger.info(f"Index loaded from {self.index_path}")
                    else:
                        logger.warning("No vectors found in index file")
            else:
                logger.warning(f"Index file not found: {self.index_path}")
                self.vectors = None
                self.model = None

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.vectors = None
            self.model = None

    def _load_metadata(self):
        """メタデータを読み込み"""
        try:
            metadata_file = Path(self.metadata_path)
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded: {len(self.metadata)} entries")
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                self.metadata = {}

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {}

    def search_similar_faces(self, query_vector: np.ndarray, k: int = None) -> List[Dict]:
        """
        類似顔を検索する

        Args:
            query_vector: クエリ特徴量ベクトル
            k: 返す結果数

        Returns:
            類似顔のリスト（類似度、メタデータ含む）
        """
        if k is None:
            k = settings.similarity_search_k

        if self.model is None or self.vectors is None or len(self.vectors) == 0:
            logger.warning("No index available for search")
            return []

        try:
            # ベクトルの正規化
            query_vector = query_vector.astype(np.float32)
            if np.linalg.norm(query_vector) > 0:
                query_vector = query_vector / np.linalg.norm(query_vector)

            # 検索実行
            query_vector = query_vector.reshape(1, -1)
            k = min(k, len(self.vectors))
            distances, indices = self.model.kneighbors(query_vector, n_neighbors=k)

            # コサイン類似度を計算（距離から変換）
            similarities = 1 - distances[0]

            # 結果を整理
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities, indices[0])):
                # 類似度が閾値以下の場合はスキップ
                if similarity < settings.similarity_threshold:
                    continue

                # メタデータを取得
                metadata = self.metadata.get(str(idx), {})

                result = {
                    "rank": i + 1,
                    "similarity": float(similarity),
                    "index": int(idx),
                    "image_id": metadata.get("image_id", f"unknown_{idx}"),
                    "image_url": metadata.get("image_url", ""),
                    "person_name": metadata.get("person_name", "不明"),
                    "era": metadata.get("era", "不明"),
                    "source": metadata.get("source", ""),
                    "collection": metadata.get("collection", ""),
                    "license": metadata.get("license", "CC BY 4.0"),
                    "metadata": metadata,
                }

                results.append(result)

            logger.info(f"Found {len(results)} similar faces")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def add_vector(self, vector: np.ndarray, metadata: Dict) -> int:
        """
        新しいベクトルをインデックスに追加

        Args:
            vector: 特徴量ベクトル
            metadata: メタデータ

        Returns:
            追加されたベクトルのインデックス
        """
        try:
            # ベクトルの正規化
            vector = vector.astype(np.float32)
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)

            # ベクトルリストに追加
            if self.vectors is None:
                self.vectors = [vector]
            else:
                self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])

            # インデックス
            current_size = len(self.vectors) - 1

            # メタデータを保存
            self.metadata[str(current_size)] = metadata

            # モデルを再構築
            self._rebuild_model()

            logger.debug(f"Added vector at index {current_size}")
            return current_size

        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            return -1

    def build_index(self, vectors: List[np.ndarray], metadata_list: List[Dict]):
        """
        インデックスを一括構築

        Args:
            vectors: 特徴量ベクトルのリスト
            metadata_list: メタデータのリスト
        """
        try:
            if not vectors:
                logger.warning("No vectors to build index")
                return

            logger.info(f"Building index with {len(vectors)} vectors")

            # ベクトルを正規化して配列に変換
            normalized_vectors = []
            valid_metadata = []

            for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
                if vector is not None and vector.size > 0:
                    # 正規化
                    vector = vector.astype(np.float32)
                    if np.linalg.norm(vector) > 0:
                        vector = vector / np.linalg.norm(vector)

                    normalized_vectors.append(vector)
                    valid_metadata.append(metadata)
                else:
                    logger.warning(f"Invalid vector at index {i}")

            if not normalized_vectors:
                logger.error("No valid vectors to build index")
                return

            # 配列に変換
            self.vectors = np.array(normalized_vectors).astype(np.float32)

            # メタデータを保存
            self.metadata = {}
            for i, metadata in enumerate(valid_metadata):
                self.metadata[str(i)] = metadata

            # モデルを構築
            self._rebuild_model()

            logger.info(f"Index built successfully with {len(self.vectors)} vectors")

        except Exception as e:
            logger.error(f"Failed to build index: {e}")

    def _rebuild_model(self):
        """モデルを再構築"""
        if self.vectors is not None and len(self.vectors) > 0:
            k = min(settings.similarity_search_k, len(self.vectors))
            self.model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            self.model.fit(self.vectors)

    def save_index(self):
        """インデックスとメタデータを保存"""
        try:
            # ディレクトリを作成
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)

            # インデックスを保存
            if self.vectors is not None:
                data = {"vectors": self.vectors, "model_params": {"metric": "cosine", "algorithm": "brute"}}
                with open(self.index_path, "wb") as f:
                    pickle.dump(data, f)
                logger.info(f"Index saved to {self.index_path}")

            # メタデータを保存
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved to {self.metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def get_index_size(self) -> int:
        """インデックスのサイズを取得"""
        if self.vectors is None:
            return 0
        return len(self.vectors)

    def get_vector_by_index(self, idx: int) -> Optional[np.ndarray]:
        """インデックスからベクトルを取得"""
        try:
            if self.vectors is None or idx >= len(self.vectors):
                return None
            return self.vectors[idx]

        except Exception as e:
            logger.error(f"Failed to get vector by index: {e}")
            return None

    def get_metadata_by_index(self, idx: int) -> Dict:
        """インデックスからメタデータを取得"""
        return self.metadata.get(str(idx), {})

    def search_by_metadata(self, **kwargs) -> List[Dict]:
        """
        メタデータによる検索

        Args:
            **kwargs: 検索条件（person_name, era, source等）

        Returns:
            マッチするメタデータのリスト
        """
        results = []

        for idx_str, metadata in self.metadata.items():
            match = True

            for key, value in kwargs.items():
                if key not in metadata or str(metadata[key]).lower() != str(value).lower():
                    match = False
                    break

            if match:
                result = metadata.copy()
                result["index"] = int(idx_str)
                results.append(result)

        return results

    def get_statistics(self) -> Dict:
        """インデックスの統計情報を取得"""
        stats = {
            "total_vectors": self.get_index_size(),
            "metadata_entries": len(self.metadata),
            "vector_dimension": self.vector_dimension,
            "index_type": "NearestNeighbors" if self.model else None,
        }

        # メタデータの統計
        if self.metadata:
            eras = [m.get("era", "不明") for m in self.metadata.values()]
            sources = [m.get("source", "不明") for m in self.metadata.values()]

            from collections import Counter

            stats["era_distribution"] = dict(Counter(eras))
            stats["source_distribution"] = dict(Counter(sources))

        return stats
