"""
メイン機能のテスト
"""

import pytest
from unittest.mock import patch, Mock
import argparse

from src.main import main


def test_main_with_mock_uvicorn():
    """メイン関数のテスト（uvicorn起動をモック）"""
    with patch("uvicorn.run") as mock_run:
        with patch("sys.argv", ["main.py"]):  # コマンドライン引数をモック
            main()

            # uvicorn.runが呼ばれることを確認
            mock_run.assert_called_once()

            # 呼び出し引数を確認
            args, kwargs = mock_run.call_args
            assert args[0] == "src.api:app"
            assert kwargs["host"] == "0.0.0.0"
            assert kwargs["port"] == 8000


def test_main_with_custom_args():
    """カスタム引数でのメイン関数テスト"""
    test_args = ["main.py", "--host", "127.0.0.1", "--port", "9000", "--max-images", "50"]

    with patch("uvicorn.run") as mock_run:
        with patch("sys.argv", test_args):
            main()

            # uvicorn.runが呼ばれることを確認
            mock_run.assert_called_once()

            # 呼び出し引数を確認
            args, kwargs = mock_run.call_args
            assert kwargs["host"] == "127.0.0.1"
            assert kwargs["port"] == 9000


@patch("src.kaokore_similarity_search.reset_kaokore_similarity_searcher")
def test_main_with_max_images_override(mock_reset):
    """max-images引数での設定上書きテスト"""
    test_args = ["main.py", "--max-images", "200"]

    with patch("uvicorn.run") as mock_run:
        with patch("sys.argv", test_args):
            main()

            # リセット関数が呼ばれることを確認
            mock_reset.assert_called_once()

            # 設定が更新されることを確認
            from src.config import settings

            assert settings.kaokore_max_images == 200


@patch("src.kaokore_similarity_search.reset_kaokore_similarity_searcher")
def test_main_with_max_images_zero(mock_reset):
    """max-images=0（全画像使用）のテスト"""
    test_args = ["main.py", "--max-images", "0"]

    with patch("uvicorn.run") as mock_run:
        with patch("sys.argv", test_args):
            main()

            # リセット関数が呼ばれることを確認
            mock_reset.assert_called_once()

            # 設定がNoneになることを確認（全画像使用）
            from src.config import settings

            assert settings.kaokore_max_images is None


def test_main_without_max_images():
    """max-images引数なしのテスト"""
    test_args = ["main.py"]

    with patch("uvicorn.run") as mock_run:
        with patch("src.kaokore_similarity_search.reset_kaokore_similarity_searcher") as mock_reset:
            with patch("sys.argv", test_args):
                main()

                # リセット関数が呼ばれないことを確認
                mock_reset.assert_not_called()


def test_main_debug_mode():
    """デバッグモードのテスト"""
    with patch("uvicorn.run") as mock_run:
        with patch("src.config.settings.debug", True):
            with patch("sys.argv", ["main.py"]):
                main()

                # デバッグモードでreloadが有効になることを確認
                args, kwargs = mock_run.call_args
                assert kwargs["reload"] is True


def test_main_production_mode():
    """本番モードのテスト"""
    with patch("uvicorn.run") as mock_run:
        with patch("src.config.settings.debug", False):
            with patch("sys.argv", ["main.py"]):
                main()

                # 本番モードでreloadが無効になることを確認
                args, kwargs = mock_run.call_args
                assert kwargs["reload"] is False
