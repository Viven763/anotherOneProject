#!/bin/bash
# Компактный on-start script для Vast.ai (можно вставить прямо в форму)
# ИЗМЕНИТЕ ПЕРЕМЕННЫЕ НИЖЕ!

ORCH="http://90.156.225.121:3000"
SECRET="YOUR_SECRET_KEY"
DB_URL="https://cryptoguide.tips/btcrecover-addressdbs/eth20240925.zip"
REPO=""  # Оставьте пустым или укажите GitHub URL

# === НЕ ИЗМЕНЯЙТЕ КОД НИЖЕ ===
set -e
apt-get update -qq && apt-get install -y -qq curl wget git build-essential pkg-config libssl-dev ocl-icd-opencl-dev clinfo unzip > /dev/null 2>&1
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
mkdir -p /workspace && cd /workspace
if [ -n "$REPO" ]; then git clone "$REPO" eth_recovery; else echo "No repo"; fi
cd eth_recovery 2>/dev/null || exit 1
if [ ! -f eth20240925 ]; then
    wget -q --show-progress "$DB_URL" -O eth20240925.zip
    unzip -q eth20240925.zip
    rm eth20240925.zip
fi
cargo build --release
export WORK_SERVER_URL="$ORCH" WORK_SERVER_SECRET="$SECRET" DATABASE_PATH="/workspace/eth_recovery/eth20240925"
./target/release/eth_recovery 2>&1 | tee worker.log
