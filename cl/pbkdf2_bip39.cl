// PBKDF2-HMAC-SHA512 for BIP39 Seed Generation
// Adapted from bip39-solver-gpu/cl/just_seed.cl
// BIP39 spec: 2048 iterations, salt = "mnemonic" + passphrase

// Note: copy_pad_previous and xor_seed_with_round are already defined in common.cl

// Main PBKDF2-HMAC-SHA512 function for BIP39
// mnemonic: UTF-8 encoded mnemonic phrase
// mnemonic_length: length of mnemonic in bytes
// seed: output buffer (64 bytes)
void mnemonic_to_seed(
    __generic uchar *mnemonic,
    uint mnemonic_length,
    __generic uchar *seed
) {
    // Initialize HMAC keys with IPAD and OPAD
    uchar ipad_key[128];
    uchar opad_key[128];
    
    // HMAC key handling: if key > 128 bytes, hash it first
    uchar hmac_key[128];
    uint hmac_key_len;
    
    if (mnemonic_length > 128) {
        // Hash the mnemonic to get a 64-byte key
        uchar hash_buffer[192];
        for(int i = 0; i < mnemonic_length && i < 192; i++) {
            hash_buffer[i] = mnemonic[i];
        }
        uchar hashed_key[64];
        sha512((ulong*)hash_buffer, mnemonic_length, (ulong*)hashed_key);
        for(int i = 0; i < 64; i++) {
            hmac_key[i] = hashed_key[i];
        }
        for(int i = 64; i < 128; i++) {
            hmac_key[i] = 0;
        }
        hmac_key_len = 64;
    } else {
        // Key fits in block, just copy and pad with zeros
        for(int i = 0; i < mnemonic_length; i++) {
            hmac_key[i] = mnemonic[i];
        }
        for(int i = mnemonic_length; i < 128; i++) {
            hmac_key[i] = 0;
        }
        hmac_key_len = mnemonic_length;
    }

    for(int x=0; x<128; x++) {
        ipad_key[x] = 0x36 ^ hmac_key[x];
        opad_key[x] = 0x5c ^ hmac_key[x];
    }

    // Initialize seed to zeros
    for(int x=0; x<64; x++) {
        seed[x] = 0;
    }

    uchar sha512_result[64] = { 0 };
    uchar key_previous_concat[256] = { 0 };

    // BIP39 salt: "mnemonic" + passphrase (empty by default)
    // ASCII values: m=109, n=110, e=101, m=109, o=111, n=110, i=105, c=99
    uchar salt[12] = { 109, 110, 101, 109, 111, 110, 105, 99, 0, 0, 0, 1 };

    // Prepare first round: ipad_key || salt
    for(int x=0; x<128; x++) {
        key_previous_concat[x] = ipad_key[x];
    }
    for(int x=0; x<12; x++) {
        key_previous_concat[x+128] = salt[x];
    }

    // First round of PBKDF2 - direct pointer cast to ulong* works because sha512
    // internally handles endianness with SWAP512
    sha512((ulong*)key_previous_concat, 140, (ulong*)sha512_result);
    copy_pad_previous(opad_key, sha512_result, key_previous_concat);
    sha512((ulong*)key_previous_concat, 192, (ulong*)sha512_result);
    xor_seed_with_round(seed, sha512_result);

    // Remaining 2047 iterations (BIP39 requires 2048 total)
    for(int x=1; x<2048; x++) {
        copy_pad_previous(ipad_key, sha512_result, key_previous_concat);
        sha512((ulong*)key_previous_concat, 192, (ulong*)sha512_result);
        copy_pad_previous(opad_key, sha512_result, key_previous_concat);
        sha512((ulong*)key_previous_concat, 192, (ulong*)sha512_result);
        xor_seed_with_round(seed, sha512_result);
    }
}

// Wrapper for convenience - takes string and handles length
void pbkdf2_hmac_sha512_bip39(
    uchar *password,
    uint password_len,
    uchar *salt_suffix,    // Optional passphrase (can be NULL for default)
    uint salt_suffix_len,
    uchar *output
) {
    // This is for future extensibility if we need to support passphrases
    // For now, we just use the standard BIP39 salt "mnemonic"
    mnemonic_to_seed(password, password_len, output);
}
