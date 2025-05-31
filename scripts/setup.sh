#!/bin/bash

# reki-gao セットアップスクリプト
# このスクリプトは初回セットアップを自動化します

set -e

echo "🚀 reki-gao セットアップを開始します..."

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

# Python バージョンチェック
check_python() {
    print_status "Python バージョンをチェックしています..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION が見つかりました"
        else
            print_error "Python 3.9以上が必要です。現在のバージョン: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python3 が見つかりません。Python 3.9以上をインストールしてください。"
        exit 1
    fi
}

# uv のインストールチェック
check_uv() {
    print_status "uv パッケージマネージャーをチェックしています..."
    
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version | cut -d' ' -f2)
        print_success "uv $UV_VERSION が見つかりました"
    else
        print_warning "uv が見つかりません。インストールしています..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        
        if command -v uv &> /dev/null; then
            print_success "uv のインストールが完了しました"
        else
            print_error "uv のインストールに失敗しました"
            exit 1
        fi
    fi
}

# システム依存関係のインストール
install_system_deps() {
    print_status "システム依存関係をチェックしています..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            print_status "Ubuntu/Debian 用の依存関係をインストールしています..."
            sudo apt-get update
            sudo apt-get install -y python3-dev libopencv-dev python3-opencv
        elif command -v yum &> /dev/null; then
            print_status "CentOS/RHEL 用の依存関係をインストールしています..."
            sudo yum install -y python3-devel opencv-devel
        else
            print_warning "パッケージマネージャーが見つかりません。手動でOpenCVをインストールしてください。"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            print_status "macOS 用の依存関係をインストールしています..."
            brew install opencv
        else
            print_warning "Homebrew が見つかりません。手動でOpenCVをインストールしてください。"
        fi
    else
        print_warning "未対応のOS: $OSTYPE"
        print_warning "手動でOpenCVをインストールしてください。"
    fi
}

# 仮想環境の作成
create_venv() {
    print_status "Python 仮想環境を作成しています..."
    
    if [ -d ".venv" ]; then
        print_warning "仮想環境が既に存在します。削除して再作成しますか？ (y/N)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            rm -rf .venv
            print_status "既存の仮想環境を削除しました"
        else
            print_status "既存の仮想環境を使用します"
            return
        fi
    fi
    
    uv venv
    print_success "仮想環境を作成しました"
}

# 依存関係のインストール
install_dependencies() {
    print_status "Python 依存関係をインストールしています..."
    
    # 仮想環境をアクティベート
    source .venv/bin/activate
    
    # 依存関係をインストール
    uv pip install -r requirements.txt
    
    print_success "依存関係のインストールが完了しました"
}

# 環境設定ファイルの作成
create_env_file() {
    print_status "環境設定ファイルを作成しています..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_success ".env ファイルを作成しました"
        print_warning "必要に応じて .env ファイルを編集してください"
    else
        print_warning ".env ファイルが既に存在します"
    fi
}

# 初期データのセットアップ
setup_initial_data() {
    print_status "初期データをセットアップしています..."
    
    # 仮想環境をアクティベート
    source .venv/bin/activate
    
    # 初期データセットアップを実行
    python -m src.main setup
    
    print_success "初期データのセットアップが完了しました"
}

# テストの実行
run_tests() {
    print_status "テストを実行しています..."
    
    # 仮想環境をアクティベート
    source .venv/bin/activate
    
    # テストを実行
    pytest tests/ -v
    
    if [ $? -eq 0 ]; then
        print_success "全てのテストが成功しました"
    else
        print_warning "一部のテストが失敗しました"
    fi
}

# メイン処理
main() {
    echo "======================================"
    echo "🎯 reki-gao セットアップスクリプト"
    echo "======================================"
    echo ""
    
    # 各ステップを実行
    check_python
    check_uv
    install_system_deps
    create_venv
    install_dependencies
    create_env_file
    setup_initial_data
    
    # テスト実行の確認
    echo ""
    print_status "テストを実行しますか？ (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        run_tests
    fi
    
    echo ""
    echo "======================================"
    print_success "🎉 セットアップが完了しました！"
    echo "======================================"
    echo ""
    echo "次のコマンドでAPIサーバーを起動できます："
    echo ""
    echo "  source .venv/bin/activate"
    echo "  python -m src.main"
    echo ""
    echo "または："
    echo ""
    echo "  ./scripts/start.sh"
    echo ""
    echo "APIドキュメント: http://localhost:8000/docs"
    echo ""
}

# スクリプト実行
main "$@"