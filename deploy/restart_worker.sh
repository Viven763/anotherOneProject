#!/bin/bash
# Скрипт для перезапуска worker с очисткой GPU

echo "🔄 Останавливаем все процессы eth_recovery..."
pkill -9 eth_recovery || true

echo "🧹 Очистка GPU context..."
# На NVIDIA можно попробовать сбросить GPU через nvidia-smi
nvidia-smi --gpu-reset || echo "⚠️  GPU reset не поддерживается, пропускаем"

# Альтернатива: выгрузить/загрузить модуль (требует root)
# modprobe -r nvidia_uvm && modprobe nvidia_uvm

echo "⏳ Ждем 3 секунды..."
sleep 3

echo "🚀 Запускаем worker заново..."
cd /workspace/eth_recovery || exit 1
export WORK_SERVER_URL="http://90.156.225.121:3000"
export WORK_SERVER_SECRET="15a172308d70dede515f9eecc78eaea9345b419581d0361220313d938631b12d"
export DATABASE_PATH="/workspace/eth_recovery/eth20240925"

# Запускаем с автоперезапуском при крахе
while true; do
    echo "▶️  Старт: $(date)"
    ./target/release/eth_recovery 2>&1 | tee -a worker.log
    EXIT_CODE=$?
    echo "❌ Worker упал с кодом $EXIT_CODE в $(date)"
    echo "⏳ Перезапуск через 10 секунд..."
    sleep 10
done
