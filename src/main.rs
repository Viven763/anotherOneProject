// Ethereum BIP39 Recovery Tool - GPU Worker Client
// Работает с bip39-solver-server оркестратором
// Адаптировано из bip39-solver-gpu для Ethereum + Database

mod db_loader;

use db_loader::Database;
use std::collections::HashMap;
use std::fs;
use ocl::{flags, ProQue};
use serde::Deserialize;

// === Конфигурация ===
const WORK_SERVER_URL: &str = "http://90.156.225.121:3000";
const WORK_SERVER_SECRET: &str = "15a172308d70dede515f9eecc78eaea9345b419581d0361220313d938631b12d";
const DATABASE_PATH: &str = "eth20240925";
const BATCH_SIZE: usize = 100_000; // 100K комбинаций, но local_work_size=8 для register pressure

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

    // Для Ethereum: известные слова захардкожены в kernel
    // Просто используем offset напрямую (0 до 2048^4)
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
        "pbkdf2_bip39.cl",           // ← PBKDF2-HMAC-SHA512 для BIP39
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
        "db_lookup.cl",
        "mnemonic_constants.cl",
        "mnemonic_generator.cl",
        "eth_recovery_kernel.cl",
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

    Ok(source)
}

// === GPU Worker ===

fn run_gpu_worker(db: &Database) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Запуск GPU Worker...\n");

    // 1. Build OpenCL kernel
    println!("📚 Компиляция OpenCL kernel...");
    let kernel_source = build_kernel_source()?;

    // 2. Select GPU device (prefer NVIDIA over CPU)
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

    // 3. Create OpenCL context (dims=1 как placeholder, реальный размер задается в kernel_builder)
    let pro_que = ProQue::builder()
        .src(&kernel_source)
        .dims(1) // Минимальный placeholder, не используется для kernel execution
        .platform(platform)
        .device(device)
        .build()?;

    println!("✅ OpenCL устройство: {}", pro_que.device().name()?);
    println!("   Max work group size: {}", pro_que.device().max_wg_size()?);

    // Получаем информацию о памяти GPU
    let global_mem_size = pro_que.device().info(ocl::enums::DeviceInfo::GlobalMemSize)
        .ok()
        .and_then(|info| match info {
            ocl::enums::DeviceInfoResult::GlobalMemSize(size) => Some(size as usize),
            _ => None,
        })
        .unwrap_or(8 * 1024 * 1024 * 1024); // Default 8GB if query fails

    let max_mem_alloc = pro_que.device().info(ocl::enums::DeviceInfo::MaxMemAllocSize)
        .ok()
        .and_then(|info| match info {
            ocl::enums::DeviceInfoResult::MaxMemAllocSize(size) => Some(size as usize),
            _ => None,
        })
        .unwrap_or(global_mem_size / 4); // Default to 25% of global memory

    println!("   Global memory: {} MB", global_mem_size / 1024 / 1024);
    println!("   Max allocation: {} MB", max_mem_alloc / 1024 / 1024);

    // 4. Upload database to GPU
    println!("\n📦 Загрузка БД в GPU ({} MB)...", db.stats().size_mb);
    let db_buffer = pro_que.buffer_builder()
        .len(db.records.len())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(db.get_raw_records())
        .build()?;

    println!("✅ БД загружена в GPU!\n");

    // 5. Рассчитываем оптимальный batch size на основе доступной памяти
    let db_size_bytes = db.records.len() * 12; // DbRecord = 12 bytes (4 hash + 8 addr_suffix)
    let available_memory = (global_mem_size as f64 * 0.7) as usize; // 70% от общей памяти
    let memory_for_batches = available_memory.saturating_sub(db_size_bytes);

    // Каждый work item (1 комбинация) требует:
    // - Локальные массивы в kernel: mnemonic[192], seed[64], privatekey[32]
    // - Промежуточные буферы в PBKDF2/SHA/Keccak: ~1KB стека
    // - Консервативная оценка: 2KB на work item
    let bytes_per_work_item = 2048;
    let optimal_batch_size = (memory_for_batches / bytes_per_work_item).min(BATCH_SIZE);

    println!("💾 Расчет памяти:");
    println!("   Доступно GPU памяти: {} MB", global_mem_size / 1024 / 1024);
    println!("   БД занимает: {} MB", db_size_bytes / 1024 / 1024);
    println!("   Свободно для батчей: {} MB", memory_for_batches / 1024 / 1024);
    println!("   Оптимальный batch size: {} комбинаций\n", optimal_batch_size);

    // 6. Create output buffers
    let result_mnemonic = pro_que.buffer_builder::<u8>()
        .len(192) // 24 words * 8 bytes
        .build()?;

    let result_found = pro_que.buffer_builder::<u32>()
        .len(1)
        .build()?;

    let result_offset = pro_que.buffer_builder::<u64>()
        .len(1)
        .build()?;

    println!("✅ GPU Worker готов к работе!\n");

    // Адаптивный batch size: начинаем с оптимального, уменьшаем если нехватка памяти
    let mut current_batch_size = optimal_batch_size;
    let min_batch_size = 1024; // Минимум 1024 комбинации (но local_work_size=8!)

    // 6. Main worker loop
    loop {
        // Получаем задание от оркестратора
        println!("📥 Запрос работы у оркестратора...");
        let work = match get_work() {
            Ok(w) => w,
            Err(e) => {
                eprintln!("❌ Ошибка получения работы: {}", e);
                eprintln!("   Убедитесь что оркестратор запущен на {}", WORK_SERVER_URL);
                std::thread::sleep(std::time::Duration::from_secs(5));
                continue;
            }
        };

        // Обрабатываем работу частями
        let mut processed = 0u64;
        while processed < work.batch_size {
            let chunk_size = std::cmp::min(current_batch_size as u64, work.batch_size - processed);
            let chunk_offset = work.start_offset + processed;

            println!("🔥 Chunk: offset={}, size={}", chunk_offset, chunk_size);

            // Reset found flag
            let zero = vec![0u32; 1];
            if let Err(e) = result_found.write(&zero).enq() {
                eprintln!("❌ OpenCL Error (write): {:?}", e);
                if e.to_string().contains("OUT_OF_RESOURCES") || e.to_string().contains("MEM") {
                    current_batch_size = std::cmp::max(current_batch_size / 2, min_batch_size);
                    println!("⚠️  Память: уменьшаем batch до {}", current_batch_size);
                    continue;
                }
                return Err(e.into());
            }

            // Build and execute kernel
            // ОПТИМИЗАЦИЯ: используем __local memory для больших массивов
            // Каждый поток требует 256 байт (192 mnemonic + 64 seed)
            let local_work_size = 32; // 32 потока * 256 байт = 8KB < 48KB local memory
            let scratch_size = local_work_size * 256; // Общий scratch buffer

            let kernel_result = pro_que.kernel_builder("check_mnemonics_eth_db")
                .arg(&db_buffer)
                .arg(db.records.len() as u64)
                .arg(&result_mnemonic)
                .arg(&result_found)
                .arg(&result_offset)
                .arg(chunk_offset)
                .arg_local::<u8>(scratch_size) // __local uchar scratch_memory[8KB]
                .global_work_size(chunk_size as usize)
                .local_work_size(local_work_size)
                .build()
                .and_then(|k| unsafe { k.enq() });

            if let Err(e) = kernel_result {
                eprintln!("❌ OpenCL Error (kernel): {:?}", e);
                if e.to_string().contains("OUT_OF_RESOURCES") || e.to_string().contains("MEM") {
                    current_batch_size = std::cmp::max(current_batch_size / 2, min_batch_size);
                    println!("⚠️  Память: уменьшаем batch до {}", current_batch_size);
                    continue;
                }
                return Err(e.into());
            }

            // Check if found
            let mut found = vec![0u32; 1];
            if let Err(e) = result_found.read(&mut found).enq() {
                eprintln!("❌ OpenCL Error (read): {:?}", e);
                if e.to_string().contains("OUT_OF_RESOURCES") || e.to_string().contains("MEM") {
                    current_batch_size = std::cmp::max(current_batch_size / 2, min_batch_size);
                    println!("⚠️  Память: уменьшаем batch до {}", current_batch_size);
                    continue;
                }
                return Err(e.into());
            }

            if found[0] == 1 {
                // SUCCESS!
                let mut mnemonic_bytes = vec![0u8; 192];
                result_mnemonic.read(&mut mnemonic_bytes).enq()?;

                let mut offset_vec = vec![0u64; 1];
                result_offset.read(&mut offset_vec).enq()?;

                let mnemonic = String::from_utf8_lossy(&mnemonic_bytes);
                let mnemonic_clean = mnemonic.trim_matches('\0').trim();

                // TODO: Extract ETH address from result
                let eth_address = "0x...".to_string();

                // Send to server
                log_solution(work.offset_for_server, mnemonic_clean.to_string(), eth_address)?;

                return Ok(()); // Stop after finding solution
            }

            processed += chunk_size;
            println!("   ✓ Обработано {}/{}", processed, work.batch_size);
        }

        // Mark work as complete
        println!("✅ Batch завершён, отправка подтверждения...\n");
        if let Err(e) = log_work_complete(work.offset_for_server) {
            eprintln!("⚠️  Ошибка отправки подтверждения: {}", e);
        }
    }

    Ok(())
}

// === Main ===

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ethereum BIP39 Recovery - GPU Worker ===\n");

    // 1. Информация о задаче
    println!("Задача:");
    println!("  Тип: 24-словная BIP39 мнемоника для Ethereum");
    println!("  Известно: первые 20 слов");
    println!("  Неизвестно: последние 4 слова (позиции 20-23)");
    println!("  Комбинаций: 2048^4 = 17.6 триллионов\n");

    println!("Известные слова:");
    for (i, word) in KNOWN_WORDS.iter().enumerate() {
        print!("  {:2}: {:<8}", i, word);
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    println!("\n  20-23: ???\n");

    // 2. Загружаем базу данных
    println!("📦 Загрузка базы данных адресов...");
    let db = Database::load(DATABASE_PATH)?;
    let stats = db.stats();

    println!("✅ База данных загружена:");
    println!("   Всего записей: {}", stats.total_records);
    println!("   Заполненных: {} ({:.1}%)", stats.filled_records, stats.load_factor * 100.0);
    println!("   Размер: {} MB", stats.size_mb);

    // 3. Проверяем подключение к оркестратору
    println!("\n🔗 Проверка подключения к оркестратору...");
    println!("   URL: {}", WORK_SERVER_URL);

    match reqwest::blocking::get(&format!("{}/status", WORK_SERVER_URL)) {
        Ok(_) => println!("✅ Оркестратор доступен"),
        Err(_) => {
            println!("⚠️  Оркестратор недоступен!");
            println!("   Запустите сервер: cd ../bip39-solver-server && node index.js");
            return Err("Orchestrator not available".into());
        }
    }

    // 4. Запускаем GPU worker
    run_gpu_worker(&db)?;

    Ok(())
}
