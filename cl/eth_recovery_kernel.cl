// Main Ethereum BIP39 Recovery Kernel with Database Lookup
// Adapted for 24-word mnemonic: 21 known words + 3 missing words at positions 21-23

// Database record structure (matches Rust DbRecord)
// Note: This is also defined in db_lookup.cl, but we need it here too
#ifndef DB_RECORD_T_DEFINED
#define DB_RECORD_T_DEFINED
typedef struct {
    uint hash;           // 4 bytes (big-endian)
    ulong addr_suffix;   // 8 bytes (little-endian)
} db_record_t;
#endif

// Helper: Binary search in sorted database
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
            return true;  // Found!
        }
        else if(mid_suffix < addr_suffix) {
            left = mid + 1;
        }
        else {
            if(mid == 0) break;  // Prevent underflow
            right = mid - 1;
        }
    }

    return false;
}

// Main kernel for ETH recovery - OPTIMIZED FOR LOW REGISTER USAGE
__kernel void check_mnemonics_eth_db(
    __global const db_record_t *db_table,    // Database table in GPU memory
    const ulong db_size,                     // Number of records in database
    __global uchar *result_mnemonic,          // Output: found mnemonic (192 bytes = 24*8)
    __global uint *result_found,              // Output: 1 if found, 0 otherwise
    __global ulong *result_offset,            // Output: offset/index of found mnemonic
    const ulong start_offset,                 // Starting offset for this batch
    __local uchar *scratch_memory            // Shared scratch space per work group
) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    // Check if already found (early exit)
    if(*result_found == 1) {
        return;
    }

    // Calculate current mnemonic index
    ulong current_offset = start_offset + gid;

    // Generate 24-word mnemonic from offset
    // Use local memory per-thread slice to avoid register pressure
    // Each thread gets 256 bytes of scratch space (192 mnemonic + 64 seed)
    __local uchar *mnemonic = scratch_memory + (lid * 256);

    // Hardcoded known words (positions 0-20) - 21 known, 3 unknown
    __constant const char known_21_words[21][9] = {
        "eager", "uncover", "rifle", "pluck", "bridge",
        "used", "smile", "neutral", "become", "echo",
        "habit", "wedding", "tragic", "robust", "organ",
        "inflict", "edge", "essence", "bronze", "symbol",
        "tilt"
    };

    // Initialize mnemonic area
    for(int i = 0; i < MNEMONIC_WORDS * WORD_LENGTH; i++) {
        mnemonic[i] = 0;
    }

    // Copy known words (0-20) to mnemonic with spaces
    int pos = 0;
    for(int w = 0; w < 21; w++) {
        for(int c = 0; c < 8 && known_21_words[w][c] != '\0'; c++) {
            mnemonic[pos++] = known_21_words[w][c];
        }
        mnemonic[pos++] = ' ';  // Space separator
    }

    // Calculate indices for last 3 words from current_offset WITH CHECKSUM OPTIMIZATION
    // Total combinations: 2048 × 2048 × 8 = 33,554,432 (NOT 2048^3!)
    // - w21_idx: word 22 (2048 variants)
    // - w22_idx: word 23 (2048 variants)  
    // - last_3_bits: last 3 bits of entropy (8 variants)
    // - w23_idx: word 24 calculated from checksum!
    
    uint last_3_bits = (uint)(current_offset % 8UL);                    // 0-7
    uint w22_idx = (uint)((current_offset / 8UL) % 2048UL);             // word 23
    uint w21_idx = (uint)((current_offset / 8UL / 2048UL) % 2048UL);    // word 22

    // Build word indices for first 23 words
    uint word_indices[24];
    // Copy known 21 words indices (hardcoded from wordlist)
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
    word_indices[21] = w21_idx;  // UNKNOWN
    word_indices[22] = w22_idx;  // UNKNOWN

    // Calculate word 24 with BIP39 checksum
    // Pack 253 bits (23 words × 11 bits) + 3 bits into entropy
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

    // Set last 3 bits of entropy
    entropy[31] = (entropy[31] & 0xF8) | last_3_bits;

    // SHA256 of entropy to get checksum
    uchar hash[32];
    for(int i = 0; i < 32; i++) hash[i] = 0;
    sha256_bytes(entropy, 32, hash);

    // Word 24 index = (last_3_bits << 8) | checksum_byte
    uint checksum = hash[0];
    uint w23_idx = (last_3_bits << 8) | checksum;
    word_indices[23] = w23_idx;

    // Now build mnemonic string from word indices
    pos = 0;  // Reset position
    for(int w = 0; w < 24; w++) {
        for(int c = 0; c < 8 && words[word_indices[w]][c] != '\0'; c++) {
            mnemonic[pos++] = words[word_indices[w]][c];
        }
        if(w < 23) mnemonic[pos++] = ' ';  // Space separator
    }

    // Convert mnemonic to seed using PBKDF2-HMAC-SHA512
    // Use local memory for seed too (192+64 = 256 bytes per thread)
    __local uchar *seed = scratch_memory + (lid * 256) + 192;
    for(int i = 0; i < 64; i++) seed[i] = 0;
    mnemonic_to_seed(mnemonic, MNEMONIC_WORDS * WORD_LENGTH, seed);

    // Derive Ethereum address from seed using BIP44 path m/44'/60'/0'/0/0
    uchar eth_address[20] = {0};
    derive_eth_address_bip44(seed, eth_address);

    // Lookup address in database (binary search)
    // Extract last 8 bytes for lookup
    ulong addr_suffix = 0;
    for(int i = 0; i < 8; i++) {
        addr_suffix |= ((ulong)eth_address[12 + i]) << (i * 8);
    }

    // Binary search in sorted database
    bool found = binary_search_db(db_table, db_size, addr_suffix);

    if(found) {
        // Atomic set to prevent race condition
        uint old = atomic_cmpxchg(result_found, 0, 1);

        if(old == 0) {
            // We're the first to find it!
            // Copy mnemonic to result
            for(int i = 0; i < MNEMONIC_WORDS * WORD_LENGTH; i++) {
                result_mnemonic[i] = mnemonic[i];
            }
            *result_offset = current_offset;
        }
    }
}

// Alternative kernel: Check single mnemonic with multiple derivation paths
__kernel void check_mnemonic_multiple_paths_eth_db(
    __global const uchar *mnemonic_words,         // Input mnemonic (120 bytes)
    __global const db_record_t *db_table,         // Database table
    __global uint *result_found,                  // Output: 1 if found
    __global uint *result_path_index,             // Output: which derivation path matched
    const uint num_paths                          // Number of paths to check
) {
    uint gid = get_global_id(0);

    if(gid >= num_paths || *result_found == 1) {
        return;
    }

    // Convert mnemonic to seed
    uchar seed[64] = {0};
    // TODO: mnemonic_to_seed(mnemonic_words, seed);

    // Derive address for this path index
    uchar eth_address[20] = {0};

    // Standard path: m/44'/60'/0'/0/gid
    extended_private_key_t master;
    new_master_from_seed(BITCOIN_MAINNET, seed, &master);

    extended_private_key_t purpose;
    hardened_private_child_from_private(&master, &purpose, 44);

    extended_private_key_t coin_type;
    hardened_private_child_from_private(&purpose, &coin_type, 60);

    extended_private_key_t account;
    hardened_private_child_from_private(&coin_type, &account, 0);

    extended_private_key_t change;
    normal_private_child_from_private(&account, &change, 0);

    extended_private_key_t address_key;
    normal_private_child_from_private(&change, &address_key, gid);

    extended_public_key_t pub;
    public_from_private(&address_key, &pub);

    eth_address_for_public_key(&pub, eth_address);

    // Lookup in database
    bool found = lookup_address_in_db(db_table, eth_address);

    if(found) {
        uint old = atomic_cmpxchg(result_found, 0, 1);
        if(old == 0) {
            *result_path_index = gid;
        }
    }
}
