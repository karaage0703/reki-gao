"""
KaoKoreデータセット読み込みモジュール
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import settings


class KaoKoreLoader:
    """KaoKoreデータセットローダー"""

    def __init__(self):
        self.images_dir = Path(settings.kaokore_images_dir)
        self.tags_path = Path(settings.kaokore_tags_path)
        self._metadata_cache: Optional[Dict[str, Dict[str, str]]] = None

    def load_metadata(self) -> Dict[str, Dict[str, str]]:
        """メタデータを読み込み"""
        if self._metadata_cache is not None:
            return self._metadata_cache

        metadata = {}

        if not self.tags_path.exists():
            print(f"警告: タグファイルが見つかりません: {self.tags_path}")
            return metadata

        with open(self.tags_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # ファイル名と属性を分離
                if ":" not in line:
                    continue

                filename, attributes_str = line.split(":", 1)

                # 属性をパース
                attributes = {}
                for attr in attributes_str.split(";"):
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        attributes[key] = value

                metadata[filename] = attributes

        self._metadata_cache = metadata
        return metadata

    def get_available_images(self) -> List[str]:
        """利用可能な画像ファイル一覧を取得"""
        if not self.images_dir.exists():
            print(f"警告: 画像ディレクトリが見つかりません: {self.images_dir}")
            return []

        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(self.images_dir.glob(f"*{ext}"))

        return [f.name for f in sorted(image_files)]

    def get_image_path(self, filename: str) -> Optional[Path]:
        """画像ファイルのパスを取得"""
        image_path = self.images_dir / filename
        return image_path if image_path.exists() else None

    def get_image_info(self, filename: str) -> Optional[Dict[str, str]]:
        """画像の詳細情報を取得"""
        metadata = self.load_metadata()
        return metadata.get(filename)

    def search_by_tag(self, tag: str) -> List[Tuple[str, Dict[str, str]]]:
        """タグで画像を検索"""
        metadata = self.load_metadata()
        results = []

        for filename, info in metadata.items():
            tags = info.get("タグ", "")
            if tag in tags:
                results.append((filename, info))

        return results

    def search_by_attribute(self, attribute: str, value: str) -> List[Tuple[str, Dict[str, str]]]:
        """属性で画像を検索"""
        metadata = self.load_metadata()
        results = []

        for filename, info in metadata.items():
            if info.get(attribute) == value:
                results.append((filename, info))

        return results

    def get_statistics(self) -> Dict[str, int]:
        """データセットの統計情報を取得"""
        metadata = self.load_metadata()
        stats = {
            "total_images": len(metadata),
            "gender_male": 0,
            "gender_female": 0,
            "gender_unknown": 0,
            "status_noble": 0,
            "status_warrior": 0,
            "status_commoner": 0,
            "status_unknown": 0,
        }

        for info in metadata.values():
            # 性別統計
            gender = info.get("性別", "不明")
            if gender == "男":
                stats["gender_male"] += 1
            elif gender == "女":
                stats["gender_female"] += 1
            else:
                stats["gender_unknown"] += 1

            # 身分統計
            status = info.get("身分", "不明")
            if status == "貴族":
                stats["status_noble"] += 1
            elif status == "武士":
                stats["status_warrior"] += 1
            elif status == "庶民":
                stats["status_commoner"] += 1
            else:
                stats["status_unknown"] += 1

        return stats

    def get_unique_tags(self) -> List[str]:
        """ユニークなタグ一覧を取得"""
        metadata = self.load_metadata()
        all_tags = set()

        for info in metadata.values():
            tags = info.get("タグ", "")
            if tags and tags.strip():  # 空文字列でない場合のみ処理
                # 複数のタグがセミコロンで区切られている場合
                for tag in tags.split(";"):
                    tag = tag.strip()
                    if tag:  # 空でないタグのみ追加
                        all_tags.add(tag)

        return sorted(list(all_tags))

    def get_unique_sources(self) -> List[str]:
        """ユニークな原典一覧を取得"""
        metadata = self.load_metadata()
        sources = set()

        for info in metadata.values():
            source = info.get("原典", "")
            if source:
                sources.add(source)

        return sorted(list(sources))

    def get_image_metadata(self, filename: str) -> Optional[Dict[str, str]]:
        """画像のメタデータを取得（get_image_infoのエイリアス）"""
        return self.get_image_info(filename)


# グローバルインスタンス
kaokore_loader = KaoKoreLoader()
