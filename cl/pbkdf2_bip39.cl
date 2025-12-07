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

    for(int x=0; x<128; x++) {
        ipad_key[x] = 0x36;
        opad_key[x] = 0x5c;
    }

    // XOR mnemonic with pads
    for(int x=0; x<mnemonic_length && x<128; x++) {
        ipad_key[x] = ipad_key[x] ^ mnemonic[x];
        opad_key[x] = opad_key[x] ^ mnemonic[x];
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

    // First round of PBKDF2
    sha512(key_previous_concat, 140, sha512_result);
    copy_pad_previous(opad_key, sha512_result, key_previous_concat);
    sha512(key_previous_concat, 192, sha512_result);
    xor_seed_with_round(seed, sha512_result);

    // Remaining 2047 iterations (BIP39 requires 2048 total)
    for(int x=1; x<2048; x++) {
        copy_pad_previous(ipad_key, sha512_result, key_previous_concat);
        sha512(key_previous_concat, 192, sha512_result);
        copy_pad_previous(opad_key, sha512_result, key_previous_concat);
        sha512(key_previous_concat, 192, sha512_result);
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
