// Ethereum BIP39 Recovery Tool - GPU Worker Client
// GPU генерирует адреса, CPU проверяет в БД (без загрузки БД в GPU)

mod db_loader;

use db_loader::Database;
use std::collections::HashMap;
use std::fs;
use ocl::{flags, ProQue, Buffer};
use serde::Deserialize;

// === Конфигурация ===
const WORK_SERVER_URL: &str = "http://90.156.225.121:3000";
const WORK_SERVER_SECRET: &str = "15a172308d70dede515f9eecc78eaea9345b419581d0361220313d938631b12d";
const DATABASE_PATH: &str = "eth20240925";
const BATCH_SIZE: usize = 5000000; // 256K - максимальный batch для GPU

// Известные 20 слов
const KNOWN_WORDS: [&str; 20] = [
    "switch", "over", "fever", "flavor", "real",
    "jazz", "vague", "sugar", "throw", "steak",
    "yellow", "salad", "crush", "donate", "three",
    "base", "baby", "carbon", "control", "false"
];

// === API структуры для работы с сервером ===

#[derive(Deserialize, Debug, Default)]
struct WorkResponse {
    #[serde(default)]
    indices: Vec<u128>,
    #[serde(default)]
    offset: u128,
    #[serde(default = "default_batch_size")]
    batch_size: u64,
}

fn default_batch_size() -> u64 {
    BATCH_SIZE as u64
}

struct Work {
    start_offset: u64,
    batch_size: u64,
    offset_for_server: u128,
}

// === Функции для работы с оркестратором ===

fn get_work() -> Result<Work, Box<dyn std::error::Error>> {
    let url = format!("{}/work?secret={}", WORK_SERVER_URL, WORK_SERVER_SECRET);
    let response = reqwest::blocking::get(&url)?;
    let work_response: WorkResponse = response.json()?;

    let start_offset = work_response.offset;

    Ok(Work {
        start_offset: start_offset as u64,
        batch_size: work_response.batch_size,
        offset_for_server: work_response.offset,
    })
}

fn log_work_complete(offset: u128) -> Result<(), Box<dyn std::error::Error>> {
    let mut json_body = HashMap::new();
    json_body.insert("offset", offset.to_string());
    json_body.insert("secret", WORK_SERVER_SECRET.to_string());

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/work", WORK_SERVER_URL);
    client.post(&url).json(&json_body).send()?;

    Ok(())
}

fn log_solution(offset: u128, mnemonic: String, eth_address: String) -> Result<(), Box<dyn std::error::Error>> {
    let mut json_body = HashMap::new();
    json_body.insert("mnemonic", mnemonic.clone());
    json_body.insert("eth_address", eth_address.clone());
    json_body.insert("offset", offset.to_string());
    json_body.insert("secret", WORK_SERVER_SECRET.to_string());

    let client = reqwest::blocking::Client::new();
    let url = format!("{}/mnemonic", WORK_SERVER_URL);
    client.post(&url).json(&json_body).send()?;

    println!("\n🎉🎉🎉 РЕШЕНИЕ НАЙДЕНО! 🎉🎉🎉");
    println!("Мнемоника: {}", mnemonic);
    println!("ETH адрес: {}", eth_address);
    println!("Offset: {}", offset);

    Ok(())
}

// === OpenCL Kernel Builder ===

fn build_kernel_source() -> Result<String, Box<dyn std::error::Error>> {
    let cl_dir = "cl/";

    let files = vec![
        "common.cl",
        "sha2.cl",
        "pbkdf2_bip39.cl",
        "keccak256.cl",
        "secp256k1_common.cl",
        "secp256k1_field.cl",
        "secp256k1_group.cl",
        "secp256k1_scalar.cl",
        "secp256k1_prec.cl",
        "secp256k1.cl",
        "ripemd.cl",
        "address.cl",
        "eth_address.cl",
        "mnemonic_constants.cl",
        "mnemonic_generator.cl",
        "bip39_checksum.cl",
    ];

    let mut source = String::new();

    for file in files {
        let path = format!("{}{}", cl_dir, file);
        match fs::read_to_string(&path) {
            Ok(content) => {
                source.push_str(&format!("\n// === {} ===\n", file));
                source.push_str(&content);
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Could not read {}: {}", path, e);
            }
        }
    }

    // Добавляем оптимизированный kernel с BIP39 checksum validation
    source.push_str(r#"
// === ОПТИМИЗИРОВАННЫЙ GPU Address Generator Kernel ===
// Генерирует ТОЛЬКО валидные BIP39 мнемоники с правильным checksum
// Оптимизация: 2048^3 комбинаций вместо 2048^4 (в 256 раз быстрее!)

__kernel void generate_eth_addresses(
    __global ulong *result_addresses,     // Output: массив addr_suffix (8 bytes каждый)
    __global uchar *result_mnemonics,     // Output: массив мнемоник (192 bytes каждая)
    const ulong start_offset,             // Starting offset for this batch (0 to 2048^3-1)
    const uint batch_size                 // Количество адресов для генерации
) {
    uint gid = get_global_id(0);

    if (gid >= batch_size) {
        return;
    }

    ulong current_offset = start_offset + gid;

    // ВАЖНО: offset теперь перебирает только 2048^3 комбинаций (слова 20-22)
    // Слово 23 вычисляется из BIP39 checksum

    // Calculate indices for words 20-22 (only 3 words, NOT 4!)
    uint w22_idx = (uint)(current_offset % 2048UL);          // word 23 (0-indexed as 22)
    uint w21_idx = (uint)((current_offset / 2048UL) % 2048UL);     // word 22
    uint w20_idx = (uint)((current_offset / 4194304UL) % 2048UL);  // word 21

    // Hardcoded known word indices (positions 0-19)
    __constant const uint known_indices[20] = {
        1831, 1291, 649, 655, 1424,   // switch, over, fever, flavor, real
        935, 1897, 1701, 1771, 1673,  // jazz, vague, sugar, throw, steak
        2037, 1525, 412, 522, 1768,   // yellow, salad, crush, donate, three
        136, 123, 265, 387, 636       // base, baby, carbon, control, false
    };

    // Build array of all 24 word indices
    uint word_indices[24];
    for(int i = 0; i < 20; i++) {
        word_indices[i] = known_indices[i];
    }
    word_indices[20] = w20_idx;
    word_indices[21] = w21_idx;
    word_indices[22] = w22_idx;

    // Calculate word 23 with valid BIP39 checksum
    // Pack first 256 bits (23 words * 11 bits + 3 bits from word 24)
    uchar entropy[32];
    for(int i = 0; i < 32; i++) entropy[i] = 0;

    uint bit_pos = 0;
    for(int w = 0; w < 23; w++) {
        uint word_val = word_indices[w];
        for(int b = 10; b >= 0; b--) {
            uint bit = (word_val >> b) & 1;
            uint byte_idx = bit_pos / 8;
            uint bit_idx = 7 - (bit_pos % 8);
            if(byte_idx < 32) {
                entropy[byte_idx] |= (bit << bit_idx);
            }
            bit_pos++;
        }
    }

    // Try all 8 possible values for last 3 bits and find valid checksum
    uint w23_idx = 0;
    for(uint last_3_bits = 0; last_3_bits < 8; last_3_bits++) {
        uchar temp_entropy[32];
        for(int i = 0; i < 32; i++) temp_entropy[i] = entropy[i];

        // Set bits 253-255
        temp_entropy[31] = (temp_entropy[31] & 0xF8) | last_3_bits;

        // Calculate SHA256
        uchar hash[32];
        sha256(temp_entropy, 32, hash);

        // Checksum = first 8 bits of hash
        uchar checksum = hash[0];

        // Last word = (last_3_bits << 8) | checksum
        uint candidate = (last_3_bits << 8) | checksum;

        if(candidate < 2048) {
            w23_idx = candidate;
            break;
        }
    }

    word_indices[23] = w23_idx;

    // Build mnemonic string
    uchar mnemonic[192];
    for(int i = 0; i < 192; i++) mnemonic[i] = 0;

    int pos = 0;
    for(int w = 0; w < 24; w++) {
        uint word_idx = word_indices[w];
        for(int c = 0; c < 8 && words[word_idx][c] != '\0'; c++) {
            mnemonic[pos++] = words[word_idx][c];
        }
        if(w < 23) mnemonic[pos++] = ' ';
    }

    // Convert mnemonic to seed
    uchar seed[64];
    for(int i = 0; i < 64; i++) seed[i] = 0;
    mnemonic_to_seed(mnemonic, 192, seed);

    // Derive Ethereum address
    uchar eth_address[20];
    for(int i = 0; i < 20; i++) eth_address[i] = 0;
    derive_eth_address_bip44(seed, eth_address);

    // Extract addr_suffix (last 8 bytes)
    ulong addr_suffix = 0;
    for(int i = 0; i < 8; i++) {
        addr_suffix |= ((ulong)eth_address[12 + i]) << (i * 8);
    }

    // Write results
    result_addresses[gid] = addr_suffix;

    // Copy mnemonic to output
    for(int i = 0; i < 192; i++) {
        result_mnemonics[gid * 192 + i] = mnemonic[i];
    }
}
"#);

    Ok(source)
}

// === GPU Worker ===

fn run_gpu_worker(db: &Database) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Запуск GPU Worker (CPU проверка в БД)...\n");

    // 1. Build OpenCL kernel
    println!("📚 Компиляция OpenCL kernel...");
    let kernel_source = build_kernel_source()?;

    // 2. Select GPU device
    use ocl::{Platform, Device, DeviceType};

    let platform = Platform::list()
        .into_iter()
        .find(|p| {
            p.name().unwrap_or_default().contains("NVIDIA") ||
            p.vendor().unwrap_or_default().contains("NVIDIA")
        })
        .or_else(|| Platform::list().into_iter().next())
        .ok_or("No OpenCL platform found")?;

    let device = Device::list(platform, Some(DeviceType::GPU))
        .ok()
        .and_then(|devices| devices.into_iter().next())
        .ok_or("No GPU device found")?;

    println!("📱 Выбрано устройство:");
    println!("   Platform: {}", platform.name()?);
    println!("   Device: {}", device.name()?);
    println!("   Type: GPU");

    // 3. Create OpenCL context
    let pro_que = ProQue::builder()
        .src(&kernel_source)
        .dims(1)
        .platform(platform)
        .device(device)
        .build()?;

    println!("✅ OpenCL устройство: {}", pro_que.device().name()?);
    println!("   Max work group size: {}", pro_que.device().max_wg_size()?);

    // 4. БД остаётся в RAM, не грузим в GPU!
    println!("\n💾 БД остаётся в RAM (CPU lookup)");
    println!("   Записей в БД: {}", db.records.len());
    println!("   Размер: {} MB\n", db.stats().size_mb);

    // 5. Создаём буферы для результатов GPU
    let batch_size = BATCH_SIZE;
    
    // Буфер для адресов (8 bytes * batch_size)
    let result_addresses: Buffer<u64> = pro_que.buffer_builder()
        .len(batch_size)
        .flags(flags::MEM_WRITE_ONLY)
        .build()?;

    // Буфер для мнемоник (192 bytes * batch_size)
    let result_mnemonics: Buffer<u8> = pro_que.buffer_builder()
        .len(batch_size * 192)
        .flags(flags::MEM_WRITE_ONLY)
        .build()?;

    println!("✅ GPU Worker готов! (batch_size={})\n", batch_size);

    // 6. Main worker loop
    loop {
        println!("📥 Запрос работы у оркестратора...");
        let work = match get_work() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("❌ Ошибка получения работы: {}", e);
                std::thread::sleep(std::time::Duration::from_secs(5));
                continue;
            }
        };

        let mut processed = 0u64;
        while processed < work.batch_size {
            let chunk_size = std::cmp::min(batch_size as u64, work.batch_size - processed);
            let chunk_offset = work.start_offset + processed;

            println!("🔥 GPU генерация: offset={}, size={}", chunk_offset, chunk_size);

            // Запускаем kernel
            let local_work_size = 64;
            let global_work_size = ((chunk_size as usize + local_work_size - 1) / local_work_size) * local_work_size;

            let kernel = pro_que.kernel_builder("generate_eth_addresses")
                .arg(&result_addresses)
                .arg(&result_mnemonics)
                .arg(chunk_offset)
                .arg(chunk_size as u32)
                .global_work_size(global_work_size)
                .local_work_size(local_work_size)
                .build()?;

            unsafe { kernel.enq()?; }
            pro_que.queue().finish()?;

            // Читаем результаты
            let mut addresses = vec![0u64; chunk_size as usize];
            result_addresses.read(&mut addresses).enq()?;

            let mut mnemonics_data = vec![0u8; chunk_size as usize * 192];
            result_mnemonics.read(&mut mnemonics_data).enq()?;

            // CPU проверка в БД
            print!("   🔍 CPU lookup...");
            for i in 0..chunk_size as usize {
                let addr_suffix = addresses[i];
                
                // Binary search в БД
                if db.lookup_address_suffix(addr_suffix) {
                    // НАЙДЕНО!
                    let mnemonic_start = i * 192;
                    let mnemonic_bytes = &mnemonics_data[mnemonic_start..mnemonic_start + 192];
                    let mnemonic = String::from_utf8_lossy(mnemonic_bytes);
                    let mnemonic_clean = mnemonic.trim_matches('\0').trim();
                    
                    let eth_address = format!("0x...{:016x}", addr_suffix);
                    
                    log_solution(work.offset_for_server, mnemonic_clean.to_string(), eth_address)?;
                    return Ok(());
                }
            }
            println!(" done");

            processed += chunk_size;
            println!("   ✓ Обработано {}/{}", processed, work.batch_size);
        }

        println!("✅ Batch завершён, отправка подтверждения...\n");
        if let Err(e) = log_work_complete(work.offset_for_server) {
            eprintln!("⚠️  Ошибка отправки подтверждения: {}", e);
        }
    }
}

// === Main ===

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ethereum BIP39 Recovery - GPU Worker ===\n");

    println!("Задача:");
    println!("  Тип: 24-словная BIP39 мнемоника для Ethereum");
    println!("  Известно: первые 20 слов");
    println!("  Неизвестно: последние 4 слова (позиции 20-23)");
    println!("  ");
    println!("  ⚡ ОПТИМИЗАЦИЯ: BIP39 Checksum");
    println!("  - Слова 20-22: 2048^3 комбинаций");
    println!("  - Слово 23: вычисляется из checksum");
    println!("  - Валидных комбинаций: 2048^3 = 8.59 миллиардов");
    println!("  - Это в 256 раз быстрее, чем 2048^4!\n");

    println!("Известные слова:");
    for (i, word) in KNOWN_WORDS.iter().enumerate() {
        print!("  {:2}: {:<8}", i, word);
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    println!("\n  20-23: ???\n");

    // Загружаем БД в RAM (не в GPU!)
    println!("📦 Загрузка базы данных в RAM...");
    let db = Database::load(DATABASE_PATH)?;
    let stats = db.stats();

    println!("✅ База данных загружена:");
    println!("   Всего записей: {}", stats.total_records);
    println!("   Заполненных: {} ({:.1}%)", stats.filled_records, stats.load_factor * 100.0);
    println!("   Размер: {} MB", stats.size_mb);

    // Проверяем оркестратор
    println!("\n🔗 Проверка подключения к оркестратору...");
    println!("   URL: {}", WORK_SERVER_URL);

    match reqwest::blocking::get(&format!("{}/status", WORK_SERVER_URL)) {
        Ok(_) => println!("✅ Оркестратор доступен"),
        Err(_) => {
            println!("⚠️  Оркестратор недоступен!");
            return Err("Orchestrator not available".into());
        }
    }

    // Запускаем GPU worker
    run_gpu_worker(&db)?;

    Ok(())
}
