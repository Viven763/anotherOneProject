// Database lookup for Ethereum addresses
// Database format: hash-table with 4-byte hash + 8-byte address suffix

// Database constants (from eth20240925 metadata)
#define DB_LENGTH 536870912UL        // _dbLength
#define DB_TABLE_BYTES 4294967296UL  // _table_bytes
#define ADDR_BYTES 8                  // _bytes_per_addr
#define HASH_BYTES 4                  // _hash_bytes
#define HASH_MASK 536870911U          // _hash_mask (DB_LENGTH - 1)
#define RECORD_SIZE 12                // HASH_BYTES + ADDR_BYTES

// Database record structure
// Note: Must match the definition in eth_recovery_kernel.cl and Rust's DbRecord
#ifndef DB_RECORD_T_DEFINED
#define DB_RECORD_T_DEFINED
typedef struct {
    uint hash;           // 4 bytes (big-endian)
    ulong addr_suffix;   // 8 bytes (little-endian)
} db_record_t;
#endif

// Extract last 8 bytes of Ethereum address as uint64
static inline ulong get_address_suffix(const uchar *eth_address) {
    // Ethereum address is 20 bytes, we take the last 8 bytes
    ulong suffix = 0;
    for(int i = 0; i < 8; i++) {
        suffix |= ((ulong)eth_address[12 + i]) << (i * 8);
    }
    return suffix;
}

// Calculate hash from address suffix (simple modulo for hash-table)
static inline uint calculate_hash(ulong addr_suffix) {
    return (uint)(addr_suffix & HASH_MASK);
}

// Lookup Ethereum address in database
// Returns: true if found, false otherwise
bool lookup_address_in_db(__global const db_record_t *db_table, const uchar *eth_address) {
    // Extract last 8 bytes of address
    ulong addr_suffix = get_address_suffix(eth_address);

    // Calculate initial hash position
    uint hash = calculate_hash(addr_suffix);
    uint position = hash;

    // Linear probing - try up to 16 positions
    // (empirically, most addresses should be found within 10 probes if present)
    for(int probe = 0; probe < 16; probe++) {
        // Get record at current position
        db_record_t record = db_table[position];

        // Check if slot is empty (null address)
        if(record.addr_suffix == 0UL) {
            // Empty slot found, address not in database
            return false;
        }

        // Check if this is our address
        if(record.addr_suffix == addr_suffix) {
            // FOUND!
            return true;
        }

        // Linear probing: move to next position
        position = (position + 1) & HASH_MASK;
    }

    // Not found after max probes
    return false;
}

// Alternative: Lookup with full address comparison (slower but more accurate)
// This version also checks the hash field for extra validation
bool lookup_address_in_db_strict(__global const db_record_t *db_table, const uchar *eth_address) {
    ulong addr_suffix = get_address_suffix(eth_address);
    uint expected_hash = calculate_hash(addr_suffix);
    uint position = expected_hash;

    for(int probe = 0; probe < 16; probe++) {
        db_record_t record = db_table[position];

        if(record.addr_suffix == 0UL) {
            return false;
        }

        // Check both hash and suffix
        if(record.hash == expected_hash && record.addr_suffix == addr_suffix) {
            return true;
        }

        position = (position + 1) & HASH_MASK;
    }

    return false;
}

// Batch lookup: check multiple addresses at once
// Returns: index of first found address, or -1 if none found
int lookup_addresses_batch(__global const db_record_t *db_table,
                           const uchar *eth_addresses,
                           int num_addresses) {
    for(int i = 0; i < num_addresses; i++) {
        if(lookup_address_in_db(db_table, &eth_addresses[i * 20])) {
            return i;
        }
    }
    return -1;
}

// Debug function: print address info
void print_address_info(const uchar *eth_address) {
    printf("ETH Address: 0x");
    for(int i = 0; i < 20; i++) {
        printf("%02x", eth_address[i]);
    }

    ulong suffix = get_address_suffix(eth_address);
    uint hash = calculate_hash(suffix);

    printf("\nSuffix: %016lx\n", suffix);
    printf("Hash: %08x\n", hash);
}
