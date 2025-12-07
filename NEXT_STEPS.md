# Следующие шаги для завершения проекта

## 🎯 Текущий статус: 95% готовности

**Осталось**: Добавить PBKDF2-HMAC-SHA512 implementation для функции `mnemonic_to_seed()`

## 📚 Найденные ресурсы

### 1. John the Ripper - PBKDF2-HMAC-SHA512 OpenCL Kernel
**Ссылка**: [pbkdf2_hmac_sha512_kernel.cl](https://github.com/openwall/john/blob/bleeding-jumbo/run/opencl/pbkdf2_hmac_sha512_kernel.cl)

**Что есть**:
- ✅ Полная реализация PBKDF2-HMAC-SHA512 для OpenCL
- ✅ Оптимизирована для GPU
- ✅ Поддержка переменного количества итераций
- ✅ Используется в John the Ripper для password cracking

**Для BIP39**:
- Нужно: 2048 iterations
- Salt: "mnemonic" + passphrase (обычно пустой)
- Input: UTF-8 NFKD normalized mnemonic
- Output: 64 bytes (512 bits)

### 2. btcrecover - SHA512 Kernel
**Ссылка**: [sha512-bc-kernel.cl](https://github.com/gurnec/btcrecover/blob/master/btcrecover/sha512-bc-kernel.cl)

**Что есть**:
- ✅ SHA512 kernel для GPU
- ✅ Оптимизирован для Bitcoin recovery
- ⚠️ Требует адаптации для PBKDF2

### 3. opencl_brute - PBKDF2 Implementation
**Ссылка**: [pbkdf2.cl](https://github.com/bkerler/opencl_brute/blob/master/Library/worker/generic/pbkdf2.cl)

**Что есть**:
- ✅ Generic PBKDF2 implementation
- ✅ Создан для BTCRecover by Stephen Rothery
- ✅ Готов к использованию

### 4. bitcoin_cracking - Optimized BIP39 Recovery
**Ссылка**: [bitcoin_cracking](https://github.com/ipsbrunoreserva/bitcoin_cracking)

**Производительность**:
- 2 million seeds/second на NVIDIA 4090 Ti
- Оптимизации: bit-masked indices, loop unrolling, vector operations

## 🔧 План действий

### Вариант 1: Использовать John the Ripper kernel (Рекомендуется) ⭐

**Плюсы**:
- Production-ready код
- Хорошо протестирован
- Оптимизирован для GPU

**Шаги**:
1. Скачать `pbkdf2_hmac_sha512_kernel.cl` из John the Ripper
2. Адаптировать под BIP39 (2048 iterations, salt="mnemonic")
3. Создать wrapper функцию `mnemonic_to_seed()`
4. Добавить в `cl/pbkdf2_bip39.cl`
5. Добавить в список загрузки kernel файлов

**Время**: ~1-2 часа

### Вариант 2: Портировать из bip39-solver-gpu

**Проверить**:
```bash
cd ../bip39-solver-gpu
grep -r "mnemonic_to_seed\|pbkdf2" src/ cl/
```

Если есть реализация там - скопировать напрямую.

**Время**: ~30 минут

### Вариант 3: Использовать готовую библиотеку

Использовать CPU версию PBKDF2 из Rust:
```rust
use pbkdf2::pbkdf2_hmac;
use sha2::Sha512;

fn mnemonic_to_seed_cpu(mnemonic: &str) -> [u8; 64] {
    let salt = format!("mnemonic{}", ""); // + passphrase
    let mut seed = [0u8; 64];
    pbkdf2_hmac::<Sha512>(
        mnemonic.as_bytes(),
        salt.as_bytes(),
        2048,
        &mut seed
    );
    seed
}
```

Затем портировать на GPU.

**Время**: ~2-3 часа

## 📝 Детальная инструкция (Вариант 1)

### Шаг 1: Скачать kernel
```bash
cd /Users/vivenlmao/Desktop/coding/myprojects/eth_recover_session/eth_recovery/cl
curl -O https://raw.githubusercontent.com/openwall/john/bleeding-jumbo/run/opencl/pbkdf2_hmac_sha512_kernel.cl
```

### Шаг 2: Создать BIP39 wrapper

Создать файл `cl/pbkdf2_bip39.cl`:

```c
// BIP39 PBKDF2-HMAC-SHA512 Wrapper
// Based on John the Ripper implementation

// Include pbkdf2_hmac_sha512_kernel.cl here
// #include "pbkdf2_hmac_sha512_kernel.cl"

// BIP39-specific wrapper
void mnemonic_to_seed(
    uchar *mnemonic,       // Input: mnemonic phrase
    uint mnemonic_len,     // Length of mnemonic
    uchar *seed            // Output: 64-byte seed
) {
    // Salt is "mnemonic" + passphrase (usually empty)
    uchar salt[9] = "mnemonic";
    uint salt_len = 8;

    // BIP39 uses 2048 iterations
    const uint iterations = 2048;

    // Call PBKDF2-HMAC-SHA512
    // This needs adaptation from John's kernel format
    // to our simpler function signature

    // TODO: Adapt pbkdf2_sha512_kernel and pbkdf2_sha512_loop
    // to work with our parameters
}
```

### Шаг 3: Обновить main.rs

Добавить в список файлов для загрузки:

```rust
let files = vec![
    "common.cl",
    "sha2.cl",
    "pbkdf2_hmac_sha512_kernel.cl",  // ← Добавить
    "pbkdf2_bip39.cl",                // ← Добавить
    "keccak256.cl",
    // ... остальные
];
```

### Шаг 4: Тестирование

Создать тест с известными данными:

```rust
// Известная мнемоника и её seed
let test_mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
let expected_seed = "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04";

// Проверить на CPU
let cpu_seed = mnemonic_to_seed_cpu(test_mnemonic);
assert_eq!(hex::encode(cpu_seed), expected_seed);

// Проверить на GPU
// ...
```

### Шаг 5: Оптимизация

После того как заработает:
1. Измерить производительность
2. Оптимизировать batch size
3. Настроить work group size
4. Benchmark на реальных данных

## 🚀 Альтернативный путь: Быстрый старт

Если хочешь максимально быстро запустить:

### 1. Используй CPU версию сначала
```bash
# В Cargo.toml уже есть pbkdf2 = "0.12"
# Просто используй её для проверки концепции
```

### 2. Запусти worker на CPU
- Скорость будет ~1000 комб/сек
- Но код будет работать и протестирован
- Можно запустить на многих CPU ядрах

### 3. Потом портируй на GPU
- Когда всё работает на CPU
- Портируешь PBKDF2 на GPU
- Получаешь x100-1000 прирост скорости

## 📊 Ожидаемая производительность

### После добавления PBKDF2:

**CPU (1 core)**:
- ~1,000 комб/сек
- Время: ~558 лет 😅

**GPU (Apple M4 Pro)**:
- ~10,000-50,000 комб/сек
- Время: ~11-56 лет

**GPU (NVIDIA RTX 4090)**:
- ~100,000-200,000 комб/сек
- Время: ~2.8 года

**100x RTX 4090**:
- Время: **~10 дней** ✅

## ✅ Чеклист перед запуском

- [ ] PBKDF2-HMAC-SHA512 реализован
- [ ] Тест с известной мнемоникой прошёл
- [ ] Kernel компилируется без ошибок
- [ ] БД загружается в GPU
- [ ] Оркестратор запущен
- [ ] Worker получает задания
- [ ] Производительность измерена
- [ ] Логирование работает
- [ ] Механизм сохранения найденных решений работает

## 🔗 Полезные ссылки

### Реализации PBKDF2-HMAC-SHA512:
- [John the Ripper - pbkdf2_hmac_sha512_kernel.cl](https://github.com/openwall/john/blob/bleeding-jumbo/run/opencl/pbkdf2_hmac_sha512_kernel.cl)
- [opencl_brute - pbkdf2.cl](https://github.com/bkerler/opencl_brute/blob/master/Library/worker/generic/pbkdf2.cl)
- [btcrecover - sha512-bc-kernel.cl](https://github.com/gurnec/btcrecover/blob/master/btcrecover/sha512-bc-kernel.cl)
- [bitcoin_cracking - GPU BIP39 Recovery Tool](https://github.com/ipsbrunoreserva/bitcoin_cracking)

### BIP39 Спецификация:
- [BIP39 Standard](https://bips.dev/39/)
- [BIP39 Tool](https://iancoleman.io/bip39/)
- [BIP39 Explanation](https://medium.com/coinmonks/mnemonic-generation-bip39-simply-explained-e9ac18db9477)

### OpenCL Optimization:
- [Acceleration Attacks on PBKDF2](https://www.usenix.org/system/files/conference/woot16/woot16-paper-ruddick.pdf)
- [John the Ripper PBKDF2 Optimization](https://github.com/magnumripper/JohnTheRipper/issues/3525)

---

**Время до запуска**: 1-3 часа (в зависимости от выбранного варианта)

**Рекомендация**: Начать с Варианта 1 (John the Ripper kernel) - самый надёжный путь! 🚀
