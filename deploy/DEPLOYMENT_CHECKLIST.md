# 🚀 Чеклист развертывания на Vast.ai

Пошаговая инструкция для запуска системы восстановления seed-фразы.

---

## 📋 Часть 1: Подготовка оркестратора (VPS)

### Шаг 1.1: Получите VPS сервер

**Рекомендуемые провайдеры:**
- DigitalOcean (от $6/месяц)
- Hetzner (от €4/месяц)
- Vultr (от $5/месяц)

**Требования:**
- OS: Ubuntu 22.04 LTS
- RAM: минимум 1 GB
- Disk: минимум 10 GB
- Публичный IP

### Шаг 1.2: Подключитесь к VPS

```bash
ssh root@YOUR_VPS_IP
```

### Шаг 1.3: Запустите установку оркестратора

```bash
# Скачайте скрипт установки
wget https://raw.githubusercontent.com/YOUR_REPO/eth_recovery/main/deploy/orchestrator_setup.sh

# Или скопируйте локальный файл
scp deploy/orchestrator_setup.sh root@YOUR_VPS_IP:/root/

# Запустите установку
chmod +x orchestrator_setup.sh
sudo bash orchestrator_setup.sh
```

**Скрипт автоматически:**
- ✅ Установит Node.js
- ✅ Создаст рабочую директорию
- ✅ Установит зависимости
- ✅ Создаст systemd сервис
- ✅ Настроит firewall
- ✅ Запустит оркестратор

### Шаг 1.4: Сохраните SECRET KEY

**ВАЖНО!** Скрипт выведет SECRET KEY в конце установки:

```
SECRET KEY: abc123def456...
```

**Сохраните его** - понадобится для workers!

### Шаг 1.5: Проверьте что оркестратор работает

```bash
# Проверка статуса сервиса
systemctl status eth-recovery-orchestrator

# Проверка API
curl http://YOUR_VPS_IP:3000/status

# Просмотр логов
tail -f /opt/eth-recovery-orchestrator/logs/orchestrator.log
```

**Ожидаемый ответ:**
```json
{
  "status": "running",
  "progress": "0.00%",
  "total": 512,
  "completed": 0,
  "in_progress": 0,
  "found": 0
}
```

---

## 📦 Часть 2: Подготовка базы данных

### Шаг 2.1: Загрузите БД на доступный сервер

**Вариант A: На том же VPS**
```bash
# На вашем Mac
scp eth_recovery/eth20240925 root@YOUR_VPS_IP:/var/www/html/

# На VPS установите веб-сервер
apt-get install -y nginx
systemctl start nginx

# БД будет доступна по:
# http://YOUR_VPS_IP/eth20240925
```

**Вариант B: Google Drive / Dropbox**
1. Загрузите файл `eth20240925` в облако
2. Получите публичную ссылку для скачивания
3. URL должен быть прямой ссылкой на файл

**Вариант C: AWS S3 / DigitalOcean Spaces**
```bash
# Загрузите в S3 bucket с публичным доступом
aws s3 cp eth20240925 s3://your-bucket/eth20240925 --acl public-read
```

### Шаг 2.2: Проверьте доступность БД

```bash
# Проверьте что файл скачивается
wget -O test_db http://YOUR_DB_URL/eth20240925
ls -lh test_db

# Ожидаемый размер: ~4.3 GB
```

---

## 🎮 Часть 3: Загрузка кода на GitHub (опционально)

Если хотите использовать GitHub для развертывания:

### Шаг 3.1: Создайте приватный репозиторий

```bash
# На вашем Mac
cd eth_recovery
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/eth_recovery.git
git push -u origin main
```

### Шаг 3.2: Создайте Personal Access Token

1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Выберите scope: `repo` (full control)
4. Скопируйте токен

**Для клонирования приватного репо используйте:**
```
https://YOUR_TOKEN@github.com/YOUR_USERNAME/eth_recovery.git
```

---

## 🌐 Часть 4: Vast.ai - Запуск workers

### Шаг 4.1: Регистрация на Vast.ai

1. Перейдите на https://vast.ai/
2. Зарегистрируйтесь
3. Пополните баланс ($10-20 для теста)

### Шаг 4.2: Поиск подходящих GPU

**Через веб-интерфейс:**
1. Search → GPU Instances
2. Фильтры:
   - GPU: RTX 3090 или RTX 4090
   - Min Reliability: 95%
   - CUDA Version: 12.x
   - Sort by: $/hr (дешевле первыми)

**Через CLI:**
```bash
# Установить Vast CLI
pip install vastai

# Войти
vastai login

# Поиск GPU
vastai search offers 'gpu_name=RTX_4090 reliability>0.95' --order 'dph+'
```

### Шаг 4.3: Подготовьте startup script

**Создайте файл `vast_startup.sh`** с вашими данными:

```bash
#!/bin/bash

# КОНФИГУРАЦИЯ - ИЗМЕНИТЕ ЭТО!
export ORCHESTRATOR_URL="http://YOUR_VPS_IP:3000"
export WORKER_SECRET="YOUR_SECRET_KEY"
export DATABASE_URL="http://YOUR_VPS_IP/eth20240925"
export GITHUB_REPO=""  # Оставьте пустым если код в образе

# Скачать и запустить основной скрипт
wget -O /tmp/startup.sh https://raw.githubusercontent.com/YOUR_REPO/eth_recovery/main/deploy/vast_worker_startup.sh
chmod +x /tmp/startup.sh
bash /tmp/startup.sh
```

Или если репозиторий приватный:

```bash
#!/bin/bash

export ORCHESTRATOR_URL="http://YOUR_VPS_IP:3000"
export WORKER_SECRET="YOUR_SECRET_KEY"
export DATABASE_URL="http://YOUR_VPS_IP/eth20240925"
export GITHUB_REPO="https://YOUR_TOKEN@github.com/YOUR_USERNAME/eth_recovery.git"

wget -O /tmp/startup.sh https://raw.githubusercontent.com/YOUR_REPO/eth_recovery/main/deploy/vast_worker_startup.sh
chmod +x /tmp/startup.sh
bash /tmp/startup.sh
```

### Шаг 4.4: Создайте инстанс

**Через веб-интерфейс:**
1. Выберите GPU и нажмите "Rent"
2. Configuration:
   - **Image**: `nvidia/cuda:12.2.0-devel-ubuntu22.04`
   - **Disk Space**: 15 GB (для кода + БД)
   - **On-start script**: Вставьте ваш `vast_startup.sh`
3. Нажмите "Rent"

**Через CLI:**
```bash
vastai create instance INSTANCE_ID \
  --image nvidia/cuda:12.2.0-devel-ubuntu22.04 \
  --disk 15 \
  --env ORCHESTRATOR_URL=http://YOUR_VPS_IP:3000 \
  --env WORKER_SECRET=YOUR_SECRET \
  --env DATABASE_URL=http://YOUR_VPS_IP/eth20240925 \
  --onstart-file vast_startup.sh
```

### Шаг 4.5: Мониторинг worker

```bash
# Получить SSH доступ
vastai ssh-url INSTANCE_ID
# Скопируйте команду SSH

# Подключитесь
ssh root@VAST_IP -p VAST_PORT

# Просмотр логов
tail -f /workspace/eth_recovery/worker.log

# Проверка GPU
nvidia-smi

# Проверка процесса
ps aux | grep eth_recovery
```

---

## 📊 Часть 5: Мониторинг и управление

### Мониторинг оркестратора

```bash
# Статус через API
curl http://YOUR_VPS_IP:3000/status

# Логи в реальном времени
ssh root@YOUR_VPS_IP
tail -f /opt/eth-recovery-orchestrator/logs/orchestrator.log
```

### Проверка прогресса

```bash
# На VPS
sqlite3 /opt/eth-recovery-orchestrator/work.db "SELECT
    COUNT(*) as total,
    SUM(CASE WHEN completed_at IS NOT NULL THEN 1 ELSE 0 END) as completed,
    ROUND(100.0 * SUM(CASE WHEN completed_at IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as progress
FROM work_queue;"
```

### Масштабирование

**Добавить больше workers:**
```bash
# Просто арендуйте еще GPU с тем же startup script
# Они автоматически подключатся к оркестратору

# Или через CLI для массового запуска
for i in {1..10}; do
    vastai create instance $INSTANCE_ID \
      --image nvidia/cuda:12.2.0-devel-ubuntu22.04 \
      --disk 15 \
      --onstart-file vast_startup.sh
    sleep 5
done
```

---

## 🎉 Часть 6: При находке решения

### Что происходит автоматически:

1. Worker находит совпадение
2. Отправляет мнемонику на оркестратор
3. Оркестратор сохраняет в БД и файл

### Как проверить:

```bash
# На VPS оркестраторе
cat /opt/eth-recovery-orchestrator/logs/FOUND_SOLUTIONS.txt

# Или через БД
sqlite3 /opt/eth-recovery-orchestrator/work.db "SELECT * FROM found_solutions;"
```

### Что делать дальше:

1. **НЕМЕДЛЕННО** импортируйте мнемонику в безопасный кошелек
2. Проверьте баланс
3. Переведите средства на новый безопасный адрес с **высоким gas price**
4. **Остановите все workers** на Vast.ai
5. Удалите логи с мнемоникой

```bash
# Остановить все Vast.ai инстансы
vastai show instances --format json | jq -r '.[].id' | xargs -I {} vastai destroy instance {}
```

---

## ⚠️ Важные замечания

### Безопасность:
- ✅ Используйте сложный SECRET KEY
- ✅ Храните мнемонику только в зашифрованном виде
- ✅ Удалите все логи после успешного восстановления
- ✅ Используйте VPN при работе с Vast.ai

### Стоимость:
- 1x RTX 4090: ~$0.35/час × 12 часов = **$4.20**
- 10x RTX 4090: ~$3.50/час × 1 час = **$3.50**
- Оркестратор VPS: ~$6/месяц
- **Общая стоимость для одного запуска: $10-15**

### Время:
- Установка оркестратора: 5 минут
- Первый worker startup: 10-15 минут (скачивание БД + компиляция)
- Последующие workers: 10-15 минут каждый
- Полный перебор на 10 GPU: ~1 час

---

## 🆘 Troubleshooting

### Оркестратор не отвечает:
```bash
systemctl status eth-recovery-orchestrator
journalctl -u eth-recovery-orchestrator -n 50
```

### Worker не подключается:
```bash
# Проверьте firewall на VPS
ufw status
ufw allow 3000/tcp

# Проверьте доступность с Vast.ai
curl http://YOUR_VPS_IP:3000/status
```

### Worker падает при компиляции:
```bash
# Увеличьте disk space до 20 GB
# Проверьте что Rust установился: rustc --version
```

### База данных не скачивается:
```bash
# Проверьте URL и размер
wget --spider http://YOUR_DB_URL/eth20240925
```

---

## ✅ Финальный чеклист

Перед запуском убедитесь:

- [ ] Оркестратор установлен и запущен на VPS
- [ ] SECRET KEY сохранен
- [ ] База данных доступна по URL
- [ ] Startup script настроен с правильными URL и SECRET
- [ ] Баланс Vast.ai пополнен ($10-20)
- [ ] Вы готовы мониторить процесс
- [ ] У вас есть план действий при находке решения

---

**Удачи в восстановлении! 🚀💰**
