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

// Известные 20 слов (4 неизвестных: позиции 20, 21, 22, 23)
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
        // "bip39_checksum.cl",  // Не используется, код встроен в kernel ниже
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
    // ДЛЯ 2 НЕИЗВЕСТНЫХ СЛОВ (22, 23) - всего 2048 × 8 = 16,384 комбинаций
    source.push_str(r#"
// === ОПТИМИЗИРОВАННЫЙ GPU Address Generator Kernel ===
// 22 известных слова + 2 неизвестных = 24 слова
// Комбинаций: 2048 × 8 = 16,384 (checksum optimization)

__kernel void generate_eth_addresses(
    __global ulong *result_addresses,     // Output: массив addr_suffix (8 bytes каждый)
    __global uchar *result_mnemonics,     // Output: массив мнемоник (192 bytes каждая)
    const ulong start_offset,             // Starting offset for this batch
    const uint batch_size                 // Количество адресов для генерации
) {
    uint gid = get_global_id(0);

    if (gid >= batch_size) {
        return;
    }

    ulong current_offset = start_offset + gid;

    // Для 2 неизвестных слов: 2048 × 8 = 16,384 комбинаций
    // - Слово 22: 2048 вариантов (11 бит)
    // - Последние 3 бита энтропии: 8 вариантов
    // - Слово 23: вычисляется из checksum

    uint last_3_bits = (uint)(current_offset % 8UL);           // 0-7
    uint w22_idx = (uint)((current_offset / 8UL) % 2048UL);    // word 23 (0-2047)

    // Build array of all 24 word indices
    // 22 известных слова (hardcoded indices from english.txt, 0-based)
    uint word_indices[24];
    word_indices[0] = 551;    // eager
    word_indices[1] = 1893;   // uncover
    word_indices[2] = 1485;   // rifle
    word_indices[3] = 1333;   // pluck
    word_indices[4] = 222;    // bridge
    word_indices[5] = 1918;   // used
    word_indices[6] = 1637;   // smile
    word_indices[7] = 1191;   // neutral
    word_indices[8] = 159;    // become
    word_indices[9] = 559;    // echo
    word_indices[10] = 833;   // habit
    word_indices[11] = 1991;  // wedding
    word_indices[12] = 1847;  // tragic
    word_indices[13] = 1498;  // robust
    word_indices[14] = 1251;  // organ
    word_indices[15] = 923;   // inflict
    word_indices[16] = 562;   // edge
    word_indices[17] = 618;   // essence
    word_indices[18] = 229;   // bronze
    word_indices[19] = 1763;  // symbol
    word_indices[20] = 1808;  // tilt
    word_indices[21] = 230;   // broom (KNOWN!)
    word_indices[22] = w22_idx;  // UNKNOWN - перебираем

    // Calculate word 23 with valid BIP39 checksum
    // Pack 253 bits (from 23 words * 11 = 253 bits)
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

    // Add last 3 bits (bits 253-255) to complete 256 bits of entropy
    entropy[31] = (entropy[31] & 0xF8) | last_3_bits;

    // Calculate SHA256 of 256-bit entropy
    uchar hash[32];
    for(int i = 0; i < 32; i++) hash[i] = 0;
    sha256_bytes(entropy, 32, hash);

    // Checksum = first 8 bits of hash
    uchar checksum = hash[0];

    // Last word (24th) = (last_3_bits << 8) | checksum
    uint w23_idx = (last_3_bits << 8) | checksum;
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
    uint mnemonic_len = pos;  // Actual length!

    // Convert mnemonic to seed
    uchar seed[64];
    for(int i = 0; i < 64; i++) seed[i] = 0;
    mnemonic_to_seed(mnemonic, mnemonic_len, seed);

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
    println!("\n🚀 Запуск GPU Worker...\n");

    println!("📚 Компиляция OpenCL kernel...");
    let kernel_source = build_kernel_source()?;

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

    let pro_que = ProQue::builder()
        .src(&kernel_source)
        .dims(1)
        .platform(platform)
        .device(device)
        .build()?;

    println!("✅ OpenCL: {}", pro_que.device().name()?);
    println!("   Max work group size: {}", pro_que.device().max_wg_size()?);

    println!("\n💾 БД в RAM (CPU lookup)");
    println!("   Записей: {}", db.records.len());
    println!("   Размер: {} MB\n", db.stats().size_mb);

    let batch_size = BATCH_SIZE;
    
    let result_addresses: Buffer<u64> = pro_que.buffer_builder()
        .len(batch_size)
        .flags(flags::MEM_WRITE_ONLY)
        .build()?;

    let result_mnemonics: Buffer<u8> = pro_que.buffer_builder()
        .len(batch_size * 192)
        .flags(flags::MEM_WRITE_ONLY)
        .build()?;

    println!("✅ GPU Worker готов! (batch_size={})\n", batch_size);

    loop {
        println!("📥 Запрос работы...");
        let work = match get_work() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("❌ Ошибка: {}", e);
                std::thread::sleep(std::time::Duration::from_secs(5));
                continue;
            }
        };

        let mut processed = 0u64;
        while processed < work.batch_size {
            let chunk_size = std::cmp::min(batch_size as u64, work.batch_size - processed);
            let chunk_offset = work.start_offset + processed;

            println!("🔥 GPU: offset={}, size={}", chunk_offset, chunk_size);

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

            let mut addresses = vec![0u64; chunk_size as usize];
            result_addresses.read(&mut addresses).enq()?;

            let mut mnemonics_data = vec![0u8; chunk_size as usize * 192];
            result_mnemonics.read(&mut mnemonics_data).enq()?;

            // DEBUG: offset 5873 (forward+cigar)
            let debug_offset = 5873u64;
            if chunk_offset <= debug_offset && debug_offset < chunk_offset + chunk_size {
                let debug_idx = (debug_offset - chunk_offset) as usize;
                let debug_mnemonic_start = debug_idx * 192;
                let debug_mnemonic_bytes = &mnemonics_data[debug_mnemonic_start..debug_mnemonic_start + 192];
                let debug_mnemonic = String::from_utf8_lossy(debug_mnemonic_bytes);
                let debug_mnemonic_clean = debug_mnemonic.trim_matches('\0').trim();
                let debug_addr = addresses[debug_idx];
                
                println!("\n🔍 DEBUG offset={}", debug_offset);
                println!("   Mnemonic: {}", debug_mnemonic_clean);
                println!("   Addr suffix: {:016x}", debug_addr);
                println!("   Expected:    fd8be886eec65fba");
                
                println!("\n📊 Первые 5:");
                for j in 0..5.min(chunk_size as usize) {
                    let m_start = j * 192;
                    let m_bytes = &mnemonics_data[m_start..m_start + 192];
                    let m = String::from_utf8_lossy(m_bytes).trim_matches('\0').trim().to_string();
                    let words: Vec<&str> = m.split_whitespace().collect();
                    let last_words = if words.len() >= 3 { 
                        format!("...{} {} {}", words[words.len()-3], words[words.len()-2], words[words.len()-1])
                    } else { 
                        m.clone() 
                    };
                    println!("   [{}] {} -> {:016x}", chunk_offset + j as u64, last_words, addresses[j]);
                }
                println!("");
            }

            print!("   🔍 CPU lookup...");
            let mut found_count = 0;
            for i in 0..chunk_size as usize {
                let addr_suffix = addresses[i];

                if db.lookup_address_suffix(addr_suffix) {
                    let mnemonic_start = i * 192;
                    let mnemonic_bytes = &mnemonics_data[mnemonic_start..mnemonic_start + 192];
                    let mnemonic = String::from_utf8_lossy(mnemonic_bytes);
                    let mnemonic_clean = mnemonic.trim_matches('\0').trim();

                    let eth_address = format!("0x...{:016x}", addr_suffix);

                    println!("\n\n🎉🎉🎉 НАЙДЕНО! 🎉🎉🎉");
                    println!("Мнемоника: {}", mnemonic_clean);
                    println!("Адрес: {}", eth_address);
                    
                    if let Err(e) = log_solution(work.offset_for_server + i as u128, mnemonic_clean.to_string(), eth_address) {
                        eprintln!("⚠️ Ошибка: {}", e);
                    }
                    found_count += 1;
                }
            }
            if found_count > 0 {
                println!(" done (найдено: {})", found_count);
            } else {
                println!(" done");
            }

            processed += chunk_size;
            println!("   ✓ {}/{}", processed, work.batch_size);
        }

        println!("✅ Batch завершён\n");
        if let Err(e) = log_work_complete(work.offset_for_server) {
            eprintln!("⚠️ Ошибка: {}", e);
        }
    }
}

// === Main ===

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set UTF-8 console on Windows
    #[cfg(windows)]
    {
        use std::process::Command;
        let _ = Command::new("cmd").args(["/C", "chcp 65001"]).output();
    }

    println!("=== Ethereum BIP39 Recovery - GPU Worker ===\n");

    println!("Задача: 24-словная BIP39 мнемоника для Ethereum");
    println!("  Известно: первые 22 слова");
    println!("  Неизвестно: последние 2 слова (позиции 22-23)");
    println!("  Checksum оптимизация: 2048 x 8 = 16,384 комбинаций\n");

    println!("Известные слова:");
    for (i, word) in KNOWN_WORDS.iter().enumerate() {
        print!("  {:2}: {:<8}", i, word);
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    println!("\n  22-23: ???\n");

    println!("📦 Загрузка базы данных в RAM...");
    let db = Database::load(DATABASE_PATH)?;
    let stats = db.stats();

    println!("✅ База данных загружена:");
    println!("   Всего записей: {}", stats.total_records);
    println!("   Заполненных: {} ({:.1}%)", stats.filled_records, stats.load_factor * 100.0);
    println!("   Размер: {} MB", stats.size_mb);

    println!("\n🔗 Проверка оркестратора...");
    println!("   URL: {}", WORK_SERVER_URL);

    match reqwest::blocking::get(&format!("{}/status", WORK_SERVER_URL)) {
        Ok(_) => println!("✅ Оркестратор доступен"),
        Err(_) => {
            println!("⚠️ Оркестратор недоступен!");
            return Err("Orchestrator not available".into());
        }
    }

    run_gpu_worker(&db)?;

    Ok(())
}
