# 🚀 Quick Start Guide - Ethereum BIP39 Recovery

Быстрый старт для развертывания системы на Vast.ai с VPS оркестратором.

---

## 📋 Что вам нужно

- [ ] VPS сервер (Ubuntu 22.04) - DigitalOcean, Hetzner, или Vultr
- [ ] Аккаунт на Vast.ai с балансом $10-20
- [ ] Доступ к файлу базы данных `eth20240925` (4.3 GB)

---

## ⚡ Быстрый старт (30 минут)

### Шаг 1: Установка оркестратора на VPS (5 минут)

```bash
# 1. Подключитесь к вашему VPS
ssh root@YOUR_VPS_IP

# 2. Скачайте скрипт установки
wget https://raw.githubusercontent.com/YOUR_REPO/eth_recovery/main/deploy/orchestrator_setup.sh

# Или скопируйте с вашего Mac:
# scp deploy/orchestrator_setup.sh root@YOUR_VPS_IP:/root/

# 3. Запустите установку
chmod +x orchestrator_setup.sh
sudo bash orchestrator_setup.sh

# 4. СОХРАНИТЕ SECRET KEY который выведет скрипт!
# Он понадобится для workers
```

**Проверка:**
```bash
curl http://YOUR_VPS_IP:3000/status
```

Должен вернуть JSON с `"status": "running"`

---

### Шаг 2: Разместите базу данных (10 минут)

**Вариант A: На том же VPS** (рекомендуется)

```bash
# На VPS установите nginx
apt-get install -y nginx
systemctl start nginx

# На вашем Mac загрузите БД
scp eth_recovery/eth20240925 root@YOUR_VPS_IP:/var/www/html/

# БД доступна по:
# http://YOUR_VPS_IP/eth20240925
```

**Вариант B: Google Drive / Dropbox**
1. Загрузите `eth20240925` в облако
2. Получите публичную ссылку для прямого скачивания

---

### Шаг 3: Запуск worker на Vast.ai (15 минут)

#### 3.1 Через веб-интерфейс Vast.ai:

1. Перейдите на https://vast.ai/
2. Console → Search GPU Instances
3. Фильтры:
   - GPU: RTX 3090 или RTX 4090
   - Min Reliability: 95%
   - CUDA: 12.x
   - Sort by: $/hr (дешевле первыми)

4. Выберите GPU и нажмите "Rent"

5. Configuration:
   - **Image**: `nvidia/cuda:12.2.0-devel-ubuntu22.04`
   - **Disk Space**: 15 GB

6. **On-start Script** - вставьте (ИЗМЕНИТЕ ПЕРЕМЕННЫЕ!):

```bash
#!/bin/bash
ORCH="http://YOUR_VPS_IP:3000"
SECRET="YOUR_SECRET_KEY"
DB_URL="http://YOUR_VPS_IP/eth20240925"
REPO=""  # Оставьте пустым если код в образе

set -e
apt-get update -qq && apt-get install -y -qq curl wget git build-essential pkg-config libssl-dev ocl-icd-opencl-dev clinfo > /dev/null 2>&1
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
mkdir -p /workspace && cd /workspace
if [ -n "$REPO" ]; then git clone "$REPO" eth_recovery; else echo "No repo"; fi
cd eth_recovery 2>/dev/null || exit 1
[ ! -f eth20240925 ] && wget -q --show-progress "$DB_URL" -O eth20240925
cargo build --release
export WORK_SERVER_URL="$ORCH" WORK_SERVER_SECRET="$SECRET" DATABASE_PATH="/workspace/eth_recovery/eth20240925"
./target/release/eth_recovery 2>&1 | tee worker.log
```

7. Нажмите "Create" и дождитесь старта

#### 3.2 Через Vast.ai CLI:

```bash
# Установить CLI
pip install vastai

# Логин
vastai login

# Найти подходящие GPU
vastai search offers 'gpu_name=RTX_4090 reliability>0.95' --order 'dph+'

# Запустить (замените OFFER_ID на ID из списка)
vastai create instance OFFER_ID \
  --image nvidia/cuda:12.2.0-devel-ubuntu22.04 \
  --disk 15 \
  --env ORCH=http://YOUR_VPS_IP:3000 \
  --env SECRET=YOUR_SECRET_KEY \
  --env DB_URL=http://YOUR_VPS_IP/eth20240925 \
  --onstart-file deploy/vast_onstart_inline.sh
```

---

## 📊 Мониторинг

### Оркестратор (VPS):

```bash
# API статус
curl http://YOUR_VPS_IP:3000/status

# Логи
ssh root@YOUR_VPS_IP
tail -f /opt/eth-recovery-orchestrator/logs/orchestrator.log

# Проверка найденных решений
cat /opt/eth-recovery-orchestrator/logs/FOUND_SOLUTIONS.txt
```

### Worker (Vast.ai):

```bash
# Получить SSH команду
vastai show instances

# Подключиться
ssh root@VAST_IP -p VAST_PORT

# Логи
tail -f /workspace/eth_recovery/worker.log

# GPU load
nvidia-smi -l 1
```

---

## 🎉 При находке решения

### Автоматически происходит:
1. Worker отправляет мнемонику на оркестратор
2. Оркестратор сохраняет в:
   - `/opt/eth-recovery-orchestrator/logs/FOUND_SOLUTIONS.txt`
   - SQLite базу данных

### Что делать:

```bash
# 1. На VPS проверьте файл
ssh root@YOUR_VPS_IP
cat /opt/eth-recovery-orchestrator/logs/FOUND_SOLUTIONS.txt

# 2. НЕМЕДЛЕННО импортируйте мнемонику в безопасный кошелек
#    (MetaMask, MyEtherWallet, etc.)

# 3. Проверьте баланс

# 4. Переведите средства на новый адрес с ВЫСОКИМ GAS PRICE

# 5. Остановите все workers
vastai destroy instance ALL_INSTANCE_IDS
```

---

## 🔧 Troubleshooting

### Оркестратор не отвечает:
```bash
ssh root@YOUR_VPS_IP
systemctl status eth-recovery-orchestrator
journalctl -u eth-recovery-orchestrator -n 50
```

### Worker не может скачать БД:
```bash
# Проверьте URL
wget --spider http://YOUR_VPS_IP/eth20240925

# Должно показать: HTTP/1.1 200 OK
```

### Worker падает при компиляции:
```bash
# Увеличьте disk space до 20 GB при создании инстанса
# Проверьте что Rust установился
ssh root@VAST_IP -p VAST_PORT
rustc --version
```

---

## 💰 Масштабирование

### Запуск нескольких workers:

**Через веб-интерфейс:**
Просто повторите Шаг 3 несколько раз с разными GPU

**Через CLI:**
```bash
# Запустить 10 workers одновременно
for i in {1..10}; do
  vastai create instance OFFER_ID \
    --image nvidia/cuda:12.2.0-devel-ubuntu22.04 \
    --disk 15 \
    --env ORCH=http://YOUR_VPS_IP:3000 \
    --env SECRET=YOUR_SECRET_KEY \
    --env DB_URL=http://YOUR_VPS_IP/eth20240925 \
    --onstart-file deploy/vast_onstart_inline.sh
  sleep 5
done
```

### Производительность:

| Workers | Скорость | Время | Стоимость |
|---------|----------|-------|-----------|
| 1x 4090 | 200k/сек | ~12ч | $4 |
| 10x 4090 | 2M/сек | ~1ч | $3.50 |
| 100x 4090 | 20M/сек | ~7мин | $4-5 |

---

## 📁 Файлы в deploy/

- `orchestrator_setup.sh` - Установка оркестратора на VPS
- `vast_worker_startup.sh` - Полный скрипт запуска worker
- `vast_onstart_inline.sh` - Компактная версия для веб-формы
- `DEPLOYMENT_CHECKLIST.md` - Детальный чеклист
- `QUICK_START.md` - Этот файл

---

## ⚠️ Важно

### Безопасность:
- ✅ Используйте сложный SECRET KEY (генерируется автоматически)
- ✅ Не загружайте приватные ключи на GitHub
- ✅ Немедленно выводите средства при находке
- ✅ Удалите все логи после успешного восстановления

### Перед запуском убедитесь:
- [ ] Оркестратор запущен и доступен по http://YOUR_VPS_IP:3000/status
- [ ] SECRET KEY сохранен
- [ ] База данных доступна для скачивания
- [ ] Баланс Vast.ai пополнен ($10-20)

---

**Готово! Можно начинать. Удачи! 🚀💰**

---

## 🆘 Нужна помощь?

См. подробную инструкцию: `DEPLOYMENT_CHECKLIST.md`
