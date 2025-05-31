#!/bin/bash

# reki-gao サーバー起動スクリプト

set -e

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 関数定義
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 使用方法の表示
show_usage() {
    echo "使用方法: $0 [OPTIONS]"
    echo ""
    echo "オプション:"
    echo "  -h, --help     このヘルプを表示"
    echo "  -d, --dev      開発モードで起動（リロード有効）"
    echo "  -p, --port     ポート番号を指定（デフォルト: 8000）"
    echo "  --host         ホストを指定（デフォルト: 0.0.0.0）"
    echo "  --setup        初期セットアップを実行してから起動"
    echo ""
    echo "例:"
    echo "  $0                    # 通常起動"
    echo "  $0 --dev             # 開発モードで起動"
    echo "  $0 --port 8080       # ポート8080で起動"
    echo "  $0 --setup           # セットアップ後に起動"
}

# デフォルト値
DEV_MODE=false
PORT=8000
HOST="0.0.0.0"
RUN_SETUP=false

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --setup)
            RUN_SETUP=true
            shift
            ;;
        *)
            print_error "未知のオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 仮想環境の確認
check_venv() {
    if [ ! -d ".venv" ]; then
        print_error "仮想環境が見つかりません。"
        print_status "セットアップスクリプトを実行してください: ./scripts/setup.sh"
        exit 1
    fi
}

# 依存関係の確認
check_dependencies() {
    print_status "依存関係をチェックしています..."
    
    source .venv/bin/activate
    
    # 主要なパッケージの確認
    python -c "import fastapi, uvicorn, opencv, numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "一部の依存関係が不足している可能性があります"
        print_status "依存関係を再インストールしています..."
        uv pip install -r requirements.txt
    fi
}

# 初期セットアップの実行
run_initial_setup() {
    if [ "$RUN_SETUP" = true ]; then
        print_status "初期セットアップを実行しています..."
        source .venv/bin/activate
        python -m src.main setup
        print_success "初期セットアップが完了しました"
    fi
}

# ヘルスチェック
health_check() {
    print_status "ヘルスチェックを実行しています..."
    
    # バックグラウンドでサーバーを起動
    source .venv/bin/activate
    python -c "
import asyncio
from src.config import ensure_directories
from src.face_detection import FaceDetector
from src.face_encoding import FaceEncoder
from src.similarity_search import SimilaritySearcher
from src.ganbo_collection import GanboCollectionManager

async def check():
    try:
        ensure_directories()
        detector = FaceDetector()
        encoder = FaceEncoder()
        searcher = SimilaritySearcher()
        manager = GanboCollectionManager()
        print('✓ 全てのコンポーネントが正常に初期化されました')
        return True
    except Exception as e:
        print(f'✗ 初期化エラー: {e}')
        return False

result = asyncio.run(check())
exit(0 if result else 1)
"
    
    if [ $? -eq 0 ]; then
        print_success "ヘルスチェックが成功しました"
    else
        print_warning "ヘルスチェックで問題が検出されました"
    fi
}

# サーバーの起動
start_server() {
    print_status "reki-gao APIサーバーを起動しています..."
    
    source .venv/bin/activate
    
    # 起動オプションの設定
    UVICORN_ARGS="src.api:app --host $HOST --port $PORT"
    
    if [ "$DEV_MODE" = true ]; then
        UVICORN_ARGS="$UVICORN_ARGS --reload --log-level debug"
        print_status "開発モードで起動します（リロード有効）"
    else
        UVICORN_ARGS="$UVICORN_ARGS --log-level info"
    fi
    
    echo ""
    echo "======================================"
    print_success "🚀 reki-gao APIサーバーを起動中..."
    echo "======================================"
    echo ""
    echo "📍 サーバー情報:"
    echo "   URL: http://$HOST:$PORT"
    echo "   API ドキュメント: http://$HOST:$PORT/docs"
    echo "   ReDoc: http://$HOST:$PORT/redoc"
    echo ""
    echo "🛑 サーバーを停止するには Ctrl+C を押してください"
    echo ""
    
    # サーバー起動
    uvicorn $UVICORN_ARGS
}

# シグナルハンドラー
cleanup() {
    echo ""
    print_status "サーバーを停止しています..."
    print_success "reki-gao APIサーバーが停止しました"
    exit 0
}

# SIGINTとSIGTERMをキャッチ
trap cleanup SIGINT SIGTERM

# メイン処理
main() {
    echo "======================================"
    echo "🎯 reki-gao サーバー起動スクリプト"
    echo "======================================"
    echo ""
    
    # 各チェックを実行
    check_venv
    check_dependencies
    run_initial_setup
    health_check
    
    # サーバー起動
    start_server
}

# スクリプト実行
main "$@"