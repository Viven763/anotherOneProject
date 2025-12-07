// Mnemonic generator for 24-word BIP39 with last 4 words missing
// Known words: positions 0-19
// Missing words: positions 20-23

// BIP39 wordlist will be passed as constant memory
// Each word is max 8 characters, stored as fixed-size array

#define WORD_LENGTH 8
#define KNOWN_WORDS 20
#define MISSING_WORDS 4
#define TOTAL_WORDS 24
#define WORDLIST_SIZE 2048

// Build mnemonic from global index
// gid maps to 4 word indices: w20, w21, w22, w23
void generate_mnemonic_24(
    ulong gid,
    __constant const uchar (*wordlist)[WORD_LENGTH],  // BIP39 wordlist
    __constant const uchar (*known_words)[WORD_LENGTH], // 20 known words
    uchar *mnemonic_output  // Output: 24 words separated by spaces
) {
    // Decompose gid into 4 word indices (each 0-2047)
    // Using bit manipulation for efficiency

    // gid ranges from 0 to 2048^4 - 1 = 17,592,186,044,415
    uint w20_idx = (uint)((gid / 8589934592UL) % 2048UL);  // gid / 2048^3
    uint w21_idx = (uint)((gid / 4194304UL) % 2048UL);     // gid / 2048^2
    uint w22_idx = (uint)((gid / 2048UL) % 2048UL);        // gid / 2048^1
    uint w23_idx = (uint)(gid % 2048UL);                   // gid % 2048

    // Clear output
    for(int i = 0; i < 200; i++) {
        mnemonic_output[i] = 0;
    }

    int pos = 0;

    // Copy known words (0-19)
    for(int w = 0; w < KNOWN_WORDS; w++) {
        // Copy word
        for(int c = 0; c < WORD_LENGTH; c++) {
            uchar ch = known_words[w][c];
            if(ch == 0) break; // End of word
            mnemonic_output[pos++] = ch;
        }
        // Add space
        mnemonic_output[pos++] = ' ';
    }

    // Add missing words (20-23)
    uint missing_indices[4] = {w20_idx, w21_idx, w22_idx, w23_idx};

    for(int w = 0; w < MISSING_WORDS; w++) {
        uint word_idx = missing_indices[w];

        // Copy word from wordlist
        for(int c = 0; c < WORD_LENGTH; c++) {
            uchar ch = wordlist[word_idx][c];
            if(ch == 0) break; // End of word
            mnemonic_output[pos++] = ch;
        }

        // Add space (except after last word)
        if(w < MISSING_WORDS - 1) {
            mnemonic_output[pos++] = ' ';
        }
    }
}

// Alternative: More efficient version that builds mnemonic in-place
void generate_mnemonic_indices(
    ulong gid,
    uint *word_indices_output  // Output: array of 24 word indices
) {
    // Decompose gid into 4 word indices
    uint w20 = (uint)((gid / 8589934592UL) % 2048UL);
    uint w21 = (uint)((gid / 4194304UL) % 2048UL);
    uint w22 = (uint)((gid / 2048UL) % 2048UL);
    uint w23 = (uint)(gid % 2048UL);

    // First 20 indices are fixed (known words)
    // They should be pre-filled or passed separately

    // Last 4 indices are generated
    word_indices_output[20] = w20;
    word_indices_output[21] = w21;
    word_indices_output[22] = w22;
    word_indices_output[23] = w23;
}

// Helper: Convert word indices to mnemonic string
void indices_to_mnemonic(
    const uint *word_indices,
    __constant const uchar (*wordlist)[WORD_LENGTH],
    uchar *mnemonic_output
) {
    int pos = 0;

    for(int w = 0; w < TOTAL_WORDS; w++) {
        uint word_idx = word_indices[w];

        // Copy word
        for(int c = 0; c < WORD_LENGTH; c++) {
            uchar ch = wordlist[word_idx][c];
            if(ch == 0) break;
            mnemonic_output[pos++] = ch;
        }

        // Add space
        if(w < TOTAL_WORDS - 1) {
            mnemonic_output[pos++] = ' ';
        }
    }
}

// Validate BIP39 checksum (last word contains checksum bits)
// For 24 words: 8 bits of checksum (from 256-bit entropy + 8-bit checksum)
bool validate_bip39_checksum_24(const uint *word_indices) {
    // Extract 11-bit values from word indices
    // 24 words * 11 bits = 264 bits total
    // = 256 bits entropy + 8 bits checksum

    uchar entropy[32] = {0}; // 256 bits

    // Pack 24 * 11-bit word indices into 264 bits
    uint bit_pos = 0;

    for(int w = 0; w < 24; w++) {
        uint word_val = word_indices[w]; // 11 bits (0-2047)

        // Write 11 bits to entropy array
        for(int b = 10; b >= 0; b--) {
            uint bit = (word_val >> b) & 1;
            uint byte_idx = bit_pos / 8;
            uint bit_idx = 7 - (bit_pos % 8);

            if(byte_idx < 32) { // Only first 256 bits
                entropy[byte_idx] |= (bit << bit_idx);
            }

            bit_pos++;
        }
    }

    // Calculate SHA256 of entropy
    uchar hash[32] = {0};
    sha256(entropy, 32, hash);

    // Extract checksum: first 8 bits of hash
    uchar computed_checksum = hash[0];

    // Extract checksum from last word (bits 256-263)
    // Last 8 bits of the 264-bit sequence
    uint last_word = word_indices[23];
    // The last word's first 3 bits are from entropy (bits 253-255)
    // The last word's last 8 bits are checksum (bits 256-263)
    // Actually, for 24 words: last 8 bits come from last word

    // Bits 253-255 (3 bits from entropy) + 256-263 (8 bits checksum) = 11 bits
    // So last word = (entropy_bits[253:255] << 8) | checksum

    // Extract checksum from mnemonic
    uchar mnemonic_checksum = (uchar)(last_word & 0xFF); // Last 8 bits

    return computed_checksum == mnemonic_checksum;
}
