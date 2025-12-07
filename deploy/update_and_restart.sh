#!/bin/bash
# Скрипт для обновления и перезапуска worker на Vast.ai
# Запускать НЕПОСРЕДСТВЕННО на сервере Vast.ai

set -e

echo "🔄 === Обновление и перезапуск GPU Worker ==="
echo ""

# 1. Остановка старых процессов
echo "1️⃣  Остановка старых процессов..."
pkill -9 eth_recovery 2>/dev/null || echo "   Процессы уже остановлены"
sleep 2

# 2. Показать состояние GPU
echo ""
echo "2️⃣  Текущее состояние GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# 3. Переход в директорию проекта
cd /workspace/eth_recovery || {
    echo "❌ Директория /workspace/eth_recovery не найдена!"
    exit 1
}

# 4. Pull последних изменений (если есть git repo)
if [ -d ".git" ]; then
    echo "3️⃣  Обновление из git..."
    git pull || echo "   ⚠️  Git pull не удался, продолжаем с локальной версией"
else
    echo "3️⃣  Git repo не найден, пропускаем pull"
fi
echo ""

# 5. Пересборка
echo "4️⃣  Пересборка проекта..."
echo "   (это займет ~10-30 секунд)"
cargo build --release 2>&1 | grep -E "(Compiling|Finished|error)" || true
echo ""

# 6. Проверка успешности сборки
if [ ! -f "./target/release/eth_recovery" ]; then
    echo "❌ Сборка не удалась! Binary не найден."
    exit 1
fi
echo "✅ Сборка успешна!"
echo ""

# 7. Запуск worker
echo "5️⃣  Запуск worker..."
export WORK_SERVER_URL="http://90.156.225.121:3000"
export WORK_SERVER_SECRET="15a172308d70dede515f9eecc78eaea9345b419581d0361220313d938631b12d"
export DATABASE_PATH="/workspace/eth_recovery/eth20240925"

# Создаем резервную копию старого лога
if [ -f "worker.log" ]; then
    mv worker.log "worker.log.$(date +%Y%m%d_%H%M%S).bak"
fi

echo ""
echo "🚀 === Запуск worker ==="
echo "   Логи записываются в: worker.log"
echo "   Для просмотра в реальном времени: tail -f worker.log"
echo ""

# Запускаем в background с автоперезапуском
nohup bash -c '
while true; do
    echo "▶️  Старт worker: $(date)" | tee -a worker.log
    ./target/release/eth_recovery 2>&1 | tee -a worker.log
    EXIT_CODE=$?
    echo "❌ Worker остановлен с кодом $EXIT_CODE: $(date)" | tee -a worker.log
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Worker завершился нормально (найдена фраза?)" | tee -a worker.log
        break
    fi
    echo "⏳ Перезапуск через 10 секунд..." | tee -a worker.log
    sleep 10
done
' > /dev/null 2>&1 &

WORKER_PID=$!
echo "✅ Worker запущен! PID: $WORKER_PID"
echo ""
echo "📊 Полезные команды:"
echo "   Логи:      tail -f worker.log"
echo "   GPU:       watch -n 1 nvidia-smi"
echo "   Процессы:  ps aux | grep eth_recovery"
echo "   Остановка: pkill -9 eth_recovery"
echo ""

# Показываем первые строки лога
sleep 3
echo "📝 Первые строки лога:"
echo "========================================"
tail -20 worker.log
