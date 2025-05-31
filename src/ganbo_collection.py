"""
顔貌コレクション管理機能
ROIS-CODH「顔貌コレクション」データの取得・処理・管理
"""

import asyncio
import aiofiles
import httpx
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
import logging
from urllib.parse import urljoin
import hashlib

from .config import settings

logger = logging.getLogger(__name__)


class GanboCollectionManager:
    """顔貌コレクション（ガンボウコレクション）データの管理クラス"""

    def __init__(self, data_dir: str = None):
        """
        初期化メソッド

        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = Path(data_dir or settings.ganbo_collection_dir)
        self.api_base_url = settings.codh_api_base_url
        self.metadata = {}

        # ディレクトリを作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images").mkdir(exist_ok=True)
        (self.data_dir / "metadata").mkdir(exist_ok=True)

        logger.info(f"GanboCollectionManager initialized with data_dir: {self.data_dir}")

    async def download_collection_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        顔貌コレクションデータをダウンロード

        Args:
            limit: ダウンロードする画像数の上限

        Returns:
            ダウンロードしたメタデータのリスト
        """
        try:
            logger.info("Starting to download Ganbo Collection data")

            # メタデータを取得
            metadata_list = await self._fetch_metadata()

            if limit:
                metadata_list = metadata_list[:limit]

            logger.info(f"Found {len(metadata_list)} items to download")

            # 画像とメタデータを並行ダウンロード
            downloaded_data = []
            semaphore = asyncio.Semaphore(5)  # 同時ダウンロード数を制限

            tasks = []
            for metadata in metadata_list:
                task = self._download_item_with_semaphore(semaphore, metadata)
                tasks.append(task)

            # 並行実行
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 成功したもののみを収集
            for result in results:
                if isinstance(result, dict):
                    downloaded_data.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Download failed: {result}")

            logger.info(f"Successfully downloaded {len(downloaded_data)} items")

            # メタデータを保存
            await self._save_metadata(downloaded_data)

            return downloaded_data

        except Exception as e:
            logger.error(f"Failed to download collection data: {e}")
            return []

    async def _fetch_metadata(self) -> List[Dict]:
        """APIからメタデータを取得"""
        try:
            # 実際のAPIエンドポイントに応じて調整が必要
            # ここではサンプルデータを生成
            logger.info("Fetching metadata from CODH API")

            # TODO: 実際のCODH APIからデータを取得
            # 現在はサンプルデータを返す
            sample_data = await self._generate_sample_metadata()

            return sample_data

        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return []

    async def _generate_sample_metadata(self) -> List[Dict]:
        """サンプルメタデータを生成（開発用）"""
        sample_data = []

        # サンプルの歴史上人物データ
        sample_persons = [
            {
                "person_name": "織田信長",
                "era": "戦国時代",
                "birth_year": "1534",
                "death_year": "1582",
                "source": "本能寺の変図屏風",
                "collection": "京都国立博物館",
                "description": "戦国時代の武将・大名",
            },
            {
                "person_name": "豊臣秀吉",
                "era": "安土桃山時代",
                "birth_year": "1537",
                "death_year": "1598",
                "source": "醍醐花見図屏風",
                "collection": "醍醐寺",
                "description": "戦国時代から安土桃山時代の武将・大名",
            },
            {
                "person_name": "徳川家康",
                "era": "江戸時代",
                "birth_year": "1543",
                "death_year": "1616",
                "source": "東照大権現像",
                "collection": "久能山東照宮",
                "description": "江戸幕府初代征夷大将軍",
            },
            {
                "person_name": "紫式部",
                "era": "平安時代",
                "birth_year": "973頃",
                "death_year": "1014頃",
                "source": "源氏物語絵巻",
                "collection": "徳川美術館",
                "description": "平安時代の女性作家・歌人",
            },
            {
                "person_name": "菅原道真",
                "era": "平安時代",
                "birth_year": "845",
                "death_year": "903",
                "source": "北野天神縁起絵巻",
                "collection": "北野天満宮",
                "description": "平安時代の貴族・学者・政治家",
            },
        ]

        for i, person in enumerate(sample_persons):
            metadata = {
                "image_id": f"ganbo_{i + 1:04d}",
                "image_url": f"https://example.com/ganbo/images/{person['person_name']}.jpg",
                "thumbnail_url": f"https://example.com/ganbo/thumbnails/{person['person_name']}_thumb.jpg",
                "person_name": person["person_name"],
                "era": person["era"],
                "birth_year": person["birth_year"],
                "death_year": person["death_year"],
                "source": person["source"],
                "collection": person["collection"],
                "description": person["description"],
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "credit": "ROIS-CODH「顔貌コレクション」",
                "face_region": {"x": 100, "y": 80, "width": 200, "height": 240},
                "image_width": 400,
                "image_height": 600,
                "file_format": "JPEG",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
            sample_data.append(metadata)

        logger.info(f"Generated {len(sample_data)} sample metadata entries")
        return sample_data

    async def _download_item_with_semaphore(self, semaphore: asyncio.Semaphore, metadata: Dict) -> Optional[Dict]:
        """セマフォを使用してアイテムをダウンロード"""
        async with semaphore:
            return await self._download_item(metadata)

    async def _download_item(self, metadata: Dict) -> Optional[Dict]:
        """個別アイテムのダウンロード"""
        try:
            image_id = metadata["image_id"]
            image_url = metadata["image_url"]

            # 画像ファイルのパス
            image_filename = f"{image_id}.jpg"
            image_path = self.data_dir / "images" / image_filename

            # 既にダウンロード済みの場合はスキップ
            if image_path.exists():
                logger.debug(f"Image already exists: {image_filename}")
                metadata["local_image_path"] = str(image_path)
                return metadata

            # 画像をダウンロード
            async with httpx.AsyncClient(timeout=30.0) as client:
                # サンプルデータの場合は実際のダウンロードをスキップ
                if "example.com" in image_url:
                    logger.debug(f"Skipping sample image download: {image_filename}")
                    # サンプル画像ファイルを作成（空ファイル）
                    async with aiofiles.open(image_path, "wb") as f:
                        await f.write(b"")
                else:
                    response = await client.get(image_url)
                    response.raise_for_status()

                    # 画像を保存
                    async with aiofiles.open(image_path, "wb") as f:
                        await f.write(response.content)

            # メタデータにローカルパスを追加
            metadata["local_image_path"] = str(image_path)
            metadata["download_timestamp"] = asyncio.get_event_loop().time()

            logger.debug(f"Downloaded: {image_filename}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to download item {metadata.get('image_id', 'unknown')}: {e}")
            return None

    async def _save_metadata(self, metadata_list: List[Dict]):
        """メタデータをファイルに保存"""
        try:
            # JSON形式で保存
            json_path = self.data_dir / "metadata" / "collection_metadata.json"
            async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(metadata_list, ensure_ascii=False, indent=2))

            # CSV形式でも保存
            csv_path = self.data_dir / "metadata" / "collection_metadata.csv"
            await self._save_metadata_csv(metadata_list, csv_path)

            logger.info(f"Metadata saved to {json_path} and {csv_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    async def _save_metadata_csv(self, metadata_list: List[Dict], csv_path: Path):
        """メタデータをCSV形式で保存"""
        if not metadata_list:
            return

        # CSVのヘッダーを取得
        headers = set()
        for metadata in metadata_list:
            headers.update(metadata.keys())
        headers = sorted(headers)

        # CSVファイルに書き込み
        async with aiofiles.open(csv_path, "w", encoding="utf-8", newline="") as f:
            # ヘッダー行
            await f.write(",".join(headers) + "\n")

            # データ行
            for metadata in metadata_list:
                row = []
                for header in headers:
                    value = metadata.get(header, "")
                    if isinstance(value, dict):
                        value = json.dumps(value, ensure_ascii=False)
                    row.append(f'"{str(value)}"')
                await f.write(",".join(row) + "\n")

    async def load_metadata(self) -> List[Dict]:
        """保存されたメタデータを読み込み"""
        try:
            json_path = self.data_dir / "metadata" / "collection_metadata.json"

            if not json_path.exists():
                logger.warning(f"Metadata file not found: {json_path}")
                return []

            async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
                content = await f.read()
                metadata_list = json.loads(content)

            logger.info(f"Loaded {len(metadata_list)} metadata entries")
            return metadata_list

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return []

    def get_metadata(self, image_id: str) -> Dict:
        """
        画像IDからメタデータを取得

        Args:
            image_id: 画像ID

        Returns:
            メタデータ辞書
        """
        return self.metadata.get(image_id, {})

    async def preprocess_images(self) -> List[Dict]:
        """
        画像の前処理と特徴量抽出の準備

        Returns:
            処理済み画像情報のリスト
        """
        try:
            # メタデータを読み込み
            metadata_list = await self.load_metadata()

            processed_images = []

            for metadata in metadata_list:
                image_path = metadata.get("local_image_path")
                if not image_path or not Path(image_path).exists():
                    logger.warning(f"Image file not found: {image_path}")
                    continue

                # 画像情報を追加
                processed_info = {
                    "image_id": metadata["image_id"],
                    "image_path": image_path,
                    "metadata": metadata,
                    "face_region": metadata.get("face_region"),
                    "ready_for_encoding": True,
                }

                processed_images.append(processed_info)

            logger.info(f"Preprocessed {len(processed_images)} images")
            return processed_images

        except Exception as e:
            logger.error(f"Failed to preprocess images: {e}")
            return []

    def get_statistics(self) -> Dict:
        """コレクションの統計情報を取得"""
        try:
            stats = {
                "total_images": 0,
                "downloaded_images": 0,
                "era_distribution": {},
                "collection_distribution": {},
                "data_directory": str(self.data_dir),
            }

            # 画像ファイル数をカウント
            images_dir = self.data_dir / "images"
            if images_dir.exists():
                stats["downloaded_images"] = len(list(images_dir.glob("*.jpg")))

            # メタデータから統計を計算
            if self.metadata:
                stats["total_images"] = len(self.metadata)

                from collections import Counter

                eras = [m.get("era", "不明") for m in self.metadata.values()]
                collections = [m.get("collection", "不明") for m in self.metadata.values()]

                stats["era_distribution"] = dict(Counter(eras))
                stats["collection_distribution"] = dict(Counter(collections))

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    async def update_collection(self):
        """コレクションデータを更新"""
        try:
            logger.info("Updating Ganbo Collection data")

            # 新しいデータをダウンロード
            new_data = await self.download_collection_data()

            # 既存のメタデータと統合
            existing_data = await self.load_metadata()
            existing_ids = {item["image_id"] for item in existing_data}

            # 新しいアイテムのみを追加
            updated_data = existing_data.copy()
            new_count = 0

            for item in new_data:
                if item["image_id"] not in existing_ids:
                    updated_data.append(item)
                    new_count += 1

            # 更新されたメタデータを保存
            await self._save_metadata(updated_data)

            logger.info(f"Collection updated: {new_count} new items added")

        except Exception as e:
            logger.error(f"Failed to update collection: {e}")
