#!/bin/bash
# Vast.ai Worker Startup Script
# Этот скрипт автоматически запускается при создании инстанса

set -e  # Exit on error

echo "=== Ethereum BIP39 Recovery - Vast.ai Worker Startup ==="
echo "Время запуска: $(date)"

# ============================================
# КОНФИГУРАЦИЯ (ИЗМЕНИТЕ ЭТИ ЗНАЧЕНИЯ!)
# ============================================

# URL вашего оркестратора (VPS)
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://YOUR_VPS_IP:3000}"

# Секретный ключ (должен совпадать с оркестратором)
WORKER_SECRET="${WORKER_SECRET:-your-secret-change-this}"

# URL для скачивания базы данных
DATABASE_URL="${DATABASE_URL:-https://cryptoguide.tips/btcrecover-addressdbs/eth20240925.zip}"

# GitHub репозиторий (если используется)
GITHUB_REPO="${GITHUB_REPO:-}"

# ============================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ============================================

echo ""
echo "📦 Установка системных зависимостей..."
apt-get update -qq
apt-get install -y -qq \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    ca-certificates \
    > /dev/null 2>&1

echo "✅ Системные зависимости установлены"

# ============================================
# УСТАНОВКА RUST
# ============================================

echo ""
echo "🦀 Установка Rust..."
if [ ! -d "$HOME/.cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "   Rust уже установлен"
fi

export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
echo "✅ Rust готов"

# ============================================
# ПРОВЕРКА GPU
# ============================================

echo ""
echo "🎮 Проверка GPU..."
nvidia-smi || echo "⚠️  nvidia-smi не найден"
clinfo || echo "⚠️  OpenCL не найден"

# ============================================
# ПОЛУЧЕНИЕ КОДА
# ============================================

echo ""
echo "📥 Получение кода проекта..."

# Создаем рабочую директорию
mkdir -p /workspace
cd /workspace

if [ -n "$GITHUB_REPO" ]; then
    # Клонируем из GitHub
    echo "   Клонирование из GitHub: $GITHUB_REPO"
    git clone "$GITHUB_REPO" eth_recovery
    cd eth_recovery
else
    # Код уже должен быть в образе или монтирован
    echo "   Используем локальный код"
    if [ ! -d "eth_recovery" ]; then
        echo "❌ Код не найден! Установите GITHUB_REPO или смонтируйте код"
        exit 1
    fi
    cd eth_recovery
fi

# ============================================
# СКАЧИВАНИЕ БАЗЫ ДАННЫХ
# ============================================

echo ""
echo "💾 Скачивание базы данных (4.3 GB)..."
echo "   URL: $DATABASE_URL"

if [ ! -f "eth20240925" ]; then
    wget -q --show-progress "$DATABASE_URL" -O eth20240925

    # Проверка размера
    FILE_SIZE=$(stat -f%z "eth20240925" 2>/dev/null || stat -c%s "eth20240925" 2>/dev/null)
    EXPECTED_SIZE=4295032832  # ~4.3 GB

    if [ "$FILE_SIZE" -lt 4000000000 ]; then
        echo "❌ База данных слишком маленькая! Ожидалось ~4.3GB, получено $(($FILE_SIZE / 1000000))MB"
        exit 1
    fi

    echo "✅ База данных скачана: $(($FILE_SIZE / 1000000))MB"
else
    echo "   База данных уже существует"
fi

# ============================================
# КОМПИЛЯЦИЯ
# ============================================

echo ""
echo "🔨 Компиляция проекта (это займет ~5-10 минут)..."
cargo build --release

if [ ! -f "target/release/eth_recovery" ]; then
    echo "❌ Компиляция не удалась!"
    exit 1
fi

echo "✅ Компиляция завершена"

# ============================================
# НАСТРОЙКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ============================================

export WORK_SERVER_URL="$ORCHESTRATOR_URL"
export WORK_SERVER_SECRET="$WORKER_SECRET"
export DATABASE_PATH="/workspace/eth_recovery/eth20240925"
export RUST_LOG=info
export RUST_BACKTRACE=1

# ============================================
# ПРОВЕРКА ПОДКЛЮЧЕНИЯ К ОРКЕСТРАТОРУ
# ============================================

echo ""
echo "🔗 Проверка подключения к оркестратору..."
echo "   URL: $ORCHESTRATOR_URL"

MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s --max-time 5 "$ORCHESTRATOR_URL/status" > /dev/null 2>&1; then
        echo "✅ Оркестратор доступен"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "   Попытка $RETRY_COUNT/$MAX_RETRIES..."
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Не удалось подключиться к оркестратору!"
    echo "   Проверьте что оркестратор запущен на: $ORCHESTRATOR_URL"
    exit 1
fi

# ============================================
# ЗАПУСК WORKER
# ============================================

echo ""
echo "🚀 Запуск GPU Worker..."
echo "============================================"
echo ""

# Запуск с логированием
exec ./target/release/eth_recovery 2>&1 | tee worker.log
