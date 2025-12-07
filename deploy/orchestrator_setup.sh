#!/bin/bash
# Скрипт установки оркестратора на VPS
# Запустите на вашем VPS сервере

set -e

echo "=== Установка Ethereum BIP39 Recovery Orchestrator ==="

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

WORK_DIR="/opt/eth-recovery-orchestrator"
SECRET_KEY="${SECRET_KEY:-$(openssl rand -hex 32)}"

# ============================================
# ПРОВЕРКА ПРАВ
# ============================================

if [ "$EUID" -ne 0 ]; then
    echo "❌ Запустите с правами root: sudo bash $0"
    exit 1
fi

# ============================================
# УСТАНОВКА NODE.JS
# ============================================

echo ""
echo "📦 Установка Node.js..."

if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

node --version
npm --version
echo "✅ Node.js установлен"

# ============================================
# СОЗДАНИЕ РАБОЧЕЙ ДИРЕКТОРИИ
# ============================================

echo ""
echo "📁 Создание рабочей директории..."

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ============================================
# СОЗДАНИЕ PACKAGE.JSON
# ============================================

cat > package.json << 'EOF'
{
  "name": "eth-recovery-orchestrator",
  "version": "1.0.0",
  "description": "BIP39 Recovery Orchestrator",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "better-sqlite3": "^9.2.2",
    "body-parser": "^1.20.2",
    "cors": "^2.8.5"
  }
}
EOF

# ============================================
# СОЗДАНИЕ INDEX.JS
# ============================================

cat > index.js << 'ENDOFJS'
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const Database = require('better-sqlite3');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const SECRET = process.env.SECRET || 'your-secret-change-this';

// Middleware
app.use(bodyParser.json());
app.use(cors());

// Логирование
const logDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir);

function log(message) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}\n`;
    console.log(logMessage.trim());
    fs.appendFileSync(path.join(logDir, 'orchestrator.log'), logMessage);
}

// База данных
const db = new Database('work.db');
db.exec(`
    CREATE TABLE IF NOT EXISTS work_queue (
        offset INTEGER PRIMARY KEY,
        batch_size INTEGER NOT NULL,
        assigned_at TEXT,
        completed_at TEXT,
        worker_id TEXT
    );

    CREATE TABLE IF NOT EXISTS found_solutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        offset INTEGER NOT NULL,
        mnemonic TEXT NOT NULL,
        eth_address TEXT NOT NULL,
        found_at TEXT NOT NULL
    );
`);

// Инициализация рабочей очереди
const TOTAL_COMBINATIONS = Math.pow(2048, 3); // 8.6 миллиардов
const BATCH_SIZE = 16777216; // 16M на задание
const TOTAL_BATCHES = Math.ceil(TOTAL_COMBINATIONS / BATCH_SIZE);

const initQueue = db.prepare(`
    INSERT OR IGNORE INTO work_queue (offset, batch_size) VALUES (?, ?)
`);

let initialized = db.prepare('SELECT COUNT(*) as count FROM work_queue').get().count;
if (initialized === 0) {
    log('Инициализация рабочей очереди...');
    const insert = db.transaction((batches) => {
        for (const batch of batches) {
            initQueue.run(batch.offset, batch.batch_size);
        }
    });

    const batches = [];
    for (let i = 0; i < TOTAL_BATCHES; i++) {
        batches.push({
            offset: i * BATCH_SIZE,
            batch_size: BATCH_SIZE
        });
    }
    insert(batches);
    log(`Создано ${TOTAL_BATCHES} заданий`);
}

// API Endpoints

// Статус оркестратора
app.get('/status', (req, res) => {
    const stats = {
        total: db.prepare('SELECT COUNT(*) as count FROM work_queue').get().count,
        completed: db.prepare('SELECT COUNT(*) as count FROM work_queue WHERE completed_at IS NOT NULL').get().count,
        in_progress: db.prepare('SELECT COUNT(*) as count FROM work_queue WHERE assigned_at IS NOT NULL AND completed_at IS NULL').get().count,
        found: db.prepare('SELECT COUNT(*) as count FROM found_solutions').get().count
    };

    const progress = ((stats.completed / stats.total) * 100).toFixed(2);

    res.json({
        status: 'running',
        progress: `${progress}%`,
        ...stats
    });
});

// Получить задание
app.get('/work', (req, res) => {
    const secret = req.query.secret;
    if (secret !== SECRET) {
        return res.status(401).json({ error: 'Invalid secret' });
    }

    // Получаем невыполненное задание
    const work = db.prepare(`
        SELECT offset, batch_size
        FROM work_queue
        WHERE completed_at IS NULL
        ORDER BY offset
        LIMIT 1
    `).get();

    if (!work) {
        return res.json({ done: true, message: 'All work completed!' });
    }

    // Отмечаем как назначенное
    const workerId = req.ip;
    db.prepare(`
        UPDATE work_queue
        SET assigned_at = datetime('now'), worker_id = ?
        WHERE offset = ?
    `).run(workerId, work.offset);

    log(`Задание выдано worker ${workerId}: offset=${work.offset}, batch=${work.batch_size}`);

    res.json({
        indices: [],
        offset: work.offset,
        batch_size: work.batch_size
    });
});

// Отметить задание как выполненное
app.post('/work', (req, res) => {
    const { offset, secret } = req.body;

    if (secret !== SECRET) {
        return res.status(401).json({ error: 'Invalid secret' });
    }

    db.prepare(`
        UPDATE work_queue
        SET completed_at = datetime('now')
        WHERE offset = ?
    `).run(offset);

    log(`Задание завершено: offset=${offset}`);
    res.json({ success: true });
});

// Сохранить найденное решение
app.post('/mnemonic', (req, res) => {
    const { mnemonic, eth_address, offset, secret } = req.body;

    if (secret !== SECRET) {
        return res.status(401).json({ error: 'Invalid secret' });
    }

    db.prepare(`
        INSERT INTO found_solutions (offset, mnemonic, eth_address, found_at)
        VALUES (?, ?, ?, datetime('now'))
    `).run(offset, mnemonic, eth_address);

    log('🎉🎉🎉 РЕШЕНИЕ НАЙДЕНО! 🎉🎉🎉');
    log(`Мнемоника: ${mnemonic}`);
    log(`ETH адрес: ${eth_address}`);
    log(`Offset: ${offset}`);

    // Сохраняем в отдельный файл
    const solutionLog = path.join(logDir, 'FOUND_SOLUTIONS.txt');
    fs.appendFileSync(solutionLog, `
========================================
Найдено: ${new Date().toISOString()}
Мнемоника: ${mnemonic}
ETH адрес: ${eth_address}
Offset: ${offset}
========================================
`);

    res.json({ success: true, message: 'Solution saved!' });
});

// Запуск сервера
app.listen(PORT, '0.0.0.0', () => {
    log(`Оркестратор запущен на порту ${PORT}`);
    log(`SECRET: ${SECRET}`);
    log(`Всего заданий: ${TOTAL_BATCHES}`);
});
ENDOFJS

# ============================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ============================================

echo ""
echo "📦 Установка npm зависимостей..."
npm install

# ============================================
# СОЗДАНИЕ SYSTEMD SERVICE
# ============================================

echo ""
echo "⚙️  Создание systemd сервиса..."

cat > /etc/systemd/system/eth-recovery-orchestrator.service << EOF
[Unit]
Description=Ethereum BIP39 Recovery Orchestrator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=bip39-solver-server
Environment="SECRET=$SECRET_KEY"
Environment="PORT=3000"
ExecStart=/usr/bin/node bip39-solver-serverindex.js
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable eth-recovery-orchestrator
systemctl start eth-recovery-orchestrator

# ============================================
# НАСТРОЙКА FIREWALL
# ============================================

echo ""
echo "🔥 Настройка firewall..."

if command -v ufw &> /dev/null; then
    ufw allow 3000/tcp
    echo "✅ Порт 3000 открыт в UFW"
fi

# ============================================
# ФИНАЛЬНАЯ ИНФОРМАЦИЯ
# ============================================

echo ""
echo "============================================"
echo "✅ Оркестратор установлен и запущен!"
echo "============================================"
echo ""
echo "📋 Информация:"
echo "   Рабочая директория: $WORK_DIR"
echo "   Порт: 3000"
echo "   SECRET KEY: $SECRET_KEY"
echo ""
echo "⚠️  ВАЖНО: Сохраните SECRET KEY!"
echo ""
echo "📊 Проверка статуса:"
echo "   systemctl status eth-recovery-orchestrator"
echo ""
echo "📝 Логи:"
echo "   journalctl -u eth-recovery-orchestrator -f"
echo "   tail -f $WORK_DIR/logs/orchestrator.log"
echo ""
echo "🌐 API:"
echo "   curl http://$(hostname -I | awk '{print $1}'):3000/status"
echo ""
echo "🔄 Управление:"
echo "   systemctl start eth-recovery-orchestrator"
echo "   systemctl stop eth-recovery-orchestrator"
echo "   systemctl restart eth-recovery-orchestrator"
echo ""
