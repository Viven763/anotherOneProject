#!/bin/bash
# Скрипт для создания systemd service на VPS
# Запустите на VPS: sudo bash create_orchestrator_service.sh

set -e

# Определяем рабочую директорию
WORK_DIR="/root/bip39-solver-server"

# Генерируем SECRET если не задан
SECRET_KEY="${SECRET_KEY:-$(openssl rand -hex 32)}"

echo "Создание systemd service для оркестратора..."
echo "Рабочая директория: $WORK_DIR"

# Создаем systemd service файл
cat > /etc/systemd/system/eth-recovery-orchestrator.service << EOF
[Unit]
Description=Ethereum BIP39 Recovery Orchestrator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$WORK_DIR
Environment="SECRET=$SECRET_KEY"
Environment="PORT=3000"
ExecStart=/usr/bin/node $WORK_DIR/index.js
Restart=always
RestartSec=10
StandardOutput=append:$WORK_DIR/logs/orchestrator.log
StandardError=append:$WORK_DIR/logs/orchestrator.log

[Install]
WantedBy=multi-user.target
EOF

# Создаем директорию для логов
mkdir -p "$WORK_DIR/logs"

# Перезагружаем systemd
systemctl daemon-reload

# Включаем автозапуск
systemctl enable eth-recovery-orchestrator

# Запускаем сервис
systemctl start eth-recovery-orchestrator

# Показываем статус
sleep 2
systemctl status eth-recovery-orchestrator --no-pager

echo ""
echo "============================================"
echo "✅ Сервис создан и запущен!"
echo "============================================"
echo ""
echo "SECRET KEY: $SECRET_KEY"
echo ""
echo "⚠️  ВАЖНО: Сохраните SECRET KEY!"
echo ""
echo "Проверка:"
echo "  systemctl status eth-recovery-orchestrator"
echo "  curl http://localhost:3000/status"
echo ""
echo "Логи:"
echo "  journalctl -u eth-recovery-orchestrator -f"
echo "  tail -f $WORK_DIR/logs/orchestrator.log"
echo ""
