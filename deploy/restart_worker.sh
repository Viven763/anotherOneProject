#!/bin/bash
# Скрипт для перезапуска worker с очисткой GPU

echo "🔄 Останавливаем все процессы eth_recovery..."
pkill -9 eth_recovery || true

echo "🧹 Очистка GPU context..."
# Пробуем сбросить через nvidia-smi (обычно не работает на потребительских GPU)
nvidia-smi --gpu-reset 2>/dev/null && echo "✅ GPU reset успешен" || echo "⚠️  GPU reset не поддерживается (это нормально)"

echo "📊 Состояние GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo "⏳ Ждем 5 секунд для очистки context..."
sleep 5

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
