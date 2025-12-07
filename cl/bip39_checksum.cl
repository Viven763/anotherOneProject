// BIP39 Checksum Calculator for 24-word mnemonics
// Оптимизация: генерируем только валидные мнемоники

// Calculate BIP39 checksum and return the last word index
// For 24 words: last word = (3 bits from entropy) | (8 bits checksum)
uint calculate_last_word_with_checksum(
    __constant const uchar (*known_words)[9], // 20 known words
    uint w20_idx,   // word 21 index (0-2047)
    uint w21_idx,   // word 22 index (0-2047)
    uint w22_idx    // word 23 index (0-2047)
) {
    // Build entropy from all 24 words:
    // - 20 known words (indices are hardcoded)
    // - 3 unknown words (w20, w21, w22)
    // - last word will be calculated

    // Each word = 11 bits
    // Total: 24 * 11 = 264 bits = 256 bits entropy + 8 bits checksum

    // We need to:
    // 1. Extract 256 bits of entropy from 23.727 words (23 words + 3 bits from word 24)
    // 2. Calculate SHA256 of these 256 bits
    // 3. Take first 8 bits of SHA256 as checksum
    // 4. Combine last 3 bits of entropy + 8 bits checksum = 11-bit word index

    // Hardcoded word indices for known words (0-19)
    // switch=1831, over=1291, fever=649, flavor=655, real=1424,
    // jazz=935, vague=1897, sugar=1701, throw=1771, steak=1673,
    // yellow=2037, salad=1525, crush=412, donate=522, three=1768,
    // base=136, baby=123, carbon=265, control=387, false=636

    __constant const uint known_indices[20] = {
        1831, 1291, 649, 655, 1424,
        935, 1897, 1701, 1771, 1673,
        2037, 1525, 412, 522, 1768,
        136, 123, 265, 387, 636
    };

    // Build array of all word indices
    uint word_indices[24];
    for(int i = 0; i < 20; i++) {
        word_indices[i] = known_indices[i];
    }
    word_indices[20] = w20_idx;
    word_indices[21] = w21_idx;
    word_indices[22] = w22_idx;
    word_indices[23] = 0; // Will be calculated

    // Pack first 256 bits into entropy array
    uchar entropy[32] = {0};
    uint bit_pos = 0;

    // Process first 23 words + 3 bits from word 24 = 256 bits total
    for(int w = 0; w < 23; w++) {
        uint word_val = word_indices[w]; // 11 bits

        // Write 11 bits to entropy
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

    // Add first 3 bits from word 24 (to complete 256 bits)
    // These 3 bits can be 0-7, giving us 8 possible last words for each (w20,w21,w22) combo
    // BUT: we want to find THE valid word, so we iterate through 0-7 and pick the one
    // where checksum matches

    // Wait, actually the problem is: we DON'T know the last 3 bits yet!
    // So we need to:
    // 1. Try all 8 possible values for last 3 bits (0-7)
    // 2. For each, calculate SHA256
    // 3. Check if resulting word index (3 bits + 8 bit checksum) is < 2048
    // 4. Return the valid one

    // Actually, BIP39 works differently:
    // - First 256 bits are random entropy
    // - Checksum = SHA256(entropy)[0:8]
    // - Last word encodes: bits 253-255 from entropy (3 bits) + checksum (8 bits)

    // So we need to try all 8 values (0-7) for the last 3 bits:
    for(uint last_3_bits = 0; last_3_bits < 8; last_3_bits++) {
        // Set bits 253-255 in entropy
        uchar temp_entropy[32];
        for(int i = 0; i < 32; i++) {
            temp_entropy[i] = entropy[i];
        }

        // Set last 3 bits (bits 253, 254, 255)
        // byte 31, bits 5,6,7
        temp_entropy[31] &= 0b11111000; // Clear last 3 bits
        temp_entropy[31] |= last_3_bits; // Set last 3 bits

        // Calculate SHA256
        uchar hash[32];
        sha256(temp_entropy, 32, hash);

        // Checksum = first 8 bits of hash
        uchar checksum = hash[0];

        // Last word index = (last_3_bits << 8) | checksum
        uint last_word_idx = (last_3_bits << 8) | checksum;

        // Validate: must be < 2048
        if(last_word_idx < 2048) {
            return last_word_idx;
        }
    }

    // This should never happen for valid BIP39
    return 0;
}
