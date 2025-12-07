// Главный OpenCL kernel для восстановления Ethereum BIP39 мнемоники
// Комбинирует все необходимые компоненты

// === 1. Константы и общие структуры ===
// Note: WORD_LENGTH and MNEMONIC_WORDS are defined in mnemonic_constants.cl
#define BIP39_WORDLIST_SIZE 2048

// Note: db_record_t is defined in eth_recovery_kernel.cl

// === 2. Include всех необходимых модулей ===

// SHA2 и PBKDF2 для BIP39
// (будет загружен из sha2.cl)

// Secp256k1 для BIP32/BIP44
// (будет загружен из secp256k1_*.cl)

// Keccak256 для Ethereum адресов
// (будет загружен из keccak256.cl)

// === 3. Генератор мнемоник ===

// Генерирует 24-словную мнемонику из gid
// gid = индекс комбинации (0 до 2048^4 - 1)
// Первые 20 слов фиксированы, последние 4 варьируются
void generate_mnemonic_24(
    ulong gid,
    __constant const uchar (*wordlist)[WORD_LENGTH],
    __constant const uchar (*known_words)[WORD_LENGTH],
    uchar *mnemonic_output
) {
    // Копируем первые 20 известных слов
    for(int i = 0; i < 20; i++) {
        for(int j = 0; j < WORD_LENGTH; j++) {
            mnemonic_output[i * WORD_LENGTH + j] = known_words[i][j];
        }
    }

    // Вычисляем индексы для последних 4 слов из gid
    // gid = w20_idx * 2048^3 + w21_idx * 2048^2 + w22_idx * 2048 + w23_idx

    ulong remaining = gid;
    uint w23_idx = (uint)(remaining % 2048UL);
    remaining /= 2048UL;

    uint w22_idx = (uint)(remaining % 2048UL);
    remaining /= 2048UL;

    uint w21_idx = (uint)(remaining % 2048UL);
    remaining /= 2048UL;

    uint w20_idx = (uint)(remaining % 2048UL);

    // Копируем последние 4 слова из wordlist
    for(int j = 0; j < WORD_LENGTH; j++) {
        mnemonic_output[20 * WORD_LENGTH + j] = wordlist[w20_idx][j];
        mnemonic_output[21 * WORD_LENGTH + j] = wordlist[w21_idx][j];
        mnemonic_output[22 * WORD_LENGTH + j] = wordlist[w22_idx][j];
        mnemonic_output[23 * WORD_LENGTH + j] = wordlist[w23_idx][j];
    }
}

// === 4. Поиск в БД ===

// Бинарный поиск addr_suffix в отсортированной БД
bool binary_search_db(
    __global const db_record_t *db_table,
    ulong db_size,
    ulong addr_suffix
) {
    ulong left = 0;
    ulong right = db_size - 1;

    while(left <= right) {
        ulong mid = left + (right - left) / 2;
        ulong mid_suffix = db_table[mid].addr_suffix;

        if(mid_suffix == addr_suffix) {
            return true;  // Найдено!
        }
        else if(mid_suffix < addr_suffix) {
            left = mid + 1;
        }
        else {
            if(mid == 0) break;  // Избегаем underflow
            right = mid - 1;
        }
    }

    return false;
}

// Получить последние 8 байт из 20-байтного ETH адреса
ulong get_address_suffix(const uchar *eth_address) {
    ulong suffix = 0;
    for(int i = 0; i < 8; i++) {
        suffix |= ((ulong)eth_address[12 + i]) << (i * 8);
    }
    return suffix;
}

// === 5. Главный kernel ===

__kernel void eth_recovery_kernel(
    ulong start_index,  // Начальный индекс для этого batch
    __constant const uchar (*wordlist)[WORD_LENGTH],
    __constant const uchar (*known_words)[WORD_LENGTH],
    __global const db_record_t *db_table,
    ulong db_size,
    __global uchar *found_flags,      // Массив флагов (1 = найдено)
    __global uchar *found_mnemonics,  // Найденные мнемоники
    __global uint *found_count        // Счётчик найденных
) {
    ulong gid = get_global_id(0);
    ulong global_index = start_index + gid;

    // Проверяем границы
    if(global_index >= 17592186044416UL) {  // 2048^4
        return;
    }

    // 1. Генерируем мнемонику
    uchar mnemonic[MNEMONIC_WORDS * WORD_LENGTH];
    generate_mnemonic_24(global_index, wordlist, known_words, mnemonic);

    // 2. Конвертируем мнемонику в seed (PBKDF2-HMAC-SHA512)
    // TODO: Реализовать через sha2.cl
    uchar seed[64];
    // mnemonic_to_seed(mnemonic, MNEMONIC_WORDS * WORD_LENGTH, seed);

    // 3. Деривация BIP32/BIP44 для Ethereum (m/44'/60'/0'/0/0)
    // TODO: Реализовать через secp256k1_*.cl
    uchar private_key[32];
    // derive_eth_key(seed, private_key);

    // 4. Получаем public key из private key
    uchar public_key[65];
    // pubkey_from_privkey(private_key, public_key);

    // 5. Генерируем ETH адрес (Keccak256 + последние 20 байт)
    // TODO: Реализовать через keccak256.cl
    uchar eth_address[20];
    // eth_address_from_pubkey(public_key, eth_address);

    // 6. Проверяем первые N адресов (как минимум первый)
    // TODO: Добавить проверку нескольких адресов по BIP44 пути

    ulong addr_suffix = get_address_suffix(eth_address);

    if(binary_search_db(db_table, db_size, addr_suffix)) {
        // НАЙДЕНО!
        uint index = atomic_inc(found_count);
        found_flags[gid] = 1;

        // Сохраняем найденную мнемонику
        for(int i = 0; i < MNEMONIC_WORDS * WORD_LENGTH; i++) {
            found_mnemonics[index * MNEMONIC_WORDS * WORD_LENGTH + i] = mnemonic[i];
        }
    }
}
