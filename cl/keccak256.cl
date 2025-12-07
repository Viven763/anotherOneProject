// Keccak-256 (SHA3) implementation for OpenCL
// Used for Ethereum address generation

#define KECCAK_ROUNDS 24

__constant ulong keccak_round_constants[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL
};

__constant uint keccak_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant uint keccak_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// Note: rotl64 is defined in sha2.cl using OpenCL's rotate() function

void keccak256(const uchar *input, uint input_len, uchar *output) {
    ulong state[25] = {0};
    uchar temp[144] = {0};

    // Absorb phase
    uint rate = 136; // 1088 bits / 8 = 136 bytes for SHA3-256
    uint offset = 0;

    while (input_len >= rate) {
        for (uint i = 0; i < rate / 8; i++) {
            ulong val = 0;
            for (uint j = 0; j < 8; j++) {
                val |= ((ulong)input[offset + i * 8 + j]) << (8 * j);
            }
            state[i] ^= val;
        }

        // Keccak-f[1600] permutation
        for (uint round = 0; round < KECCAK_ROUNDS; round++) {
            ulong C[5], D[5];

            // Theta
            for (uint i = 0; i < 5; i++) {
                C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
            }
            for (uint i = 0; i < 5; i++) {
                D[i] = C[(i + 4) % 5] ^ rotl64(C[(i + 1) % 5], 1);
            }
            for (uint i = 0; i < 25; i++) {
                state[i] ^= D[i % 5];
            }

            // Rho and Pi
            ulong B[25];
            for (uint i = 0; i < 25; i++) {
                B[keccak_piln[i]] = rotl64(state[i], keccak_rotc[i]);
            }

            // Chi
            for (uint i = 0; i < 25; i++) {
                state[i] = B[i] ^ ((~B[(i + 5) % 25]) & B[(i + 10) % 25]);
            }

            // Iota
            state[0] ^= keccak_round_constants[round];
        }

        input_len -= rate;
        offset += rate;
    }

    // Padding
    for (uint i = 0; i < 144; i++) {
        temp[i] = 0;
    }
    for (uint i = 0; i < input_len; i++) {
        temp[i] = input[offset + i];
    }
    temp[input_len] = 0x01; // Keccak padding (not SHA3 which uses 0x06)
    temp[rate - 1] |= 0x80;

    // Final block
    for (uint i = 0; i < rate / 8; i++) {
        ulong val = 0;
        for (uint j = 0; j < 8; j++) {
            val |= ((ulong)temp[i * 8 + j]) << (8 * j);
        }
        state[i] ^= val;
    }

    // Final permutation
    for (uint round = 0; round < KECCAK_ROUNDS; round++) {
        ulong C[5], D[5];

        // Theta
        for (uint i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }
        for (uint i = 0; i < 5; i++) {
            D[i] = C[(i + 4) % 5] ^ rotl64(C[(i + 1) % 5], 1);
        }
        for (uint i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        // Rho and Pi
        ulong B[25];
        for (uint i = 0; i < 25; i++) {
            B[keccak_piln[i]] = rotl64(state[i], keccak_rotc[i]);
        }

        // Chi
        for (uint i = 0; i < 25; i++) {
            state[i] = B[i] ^ ((~B[(i + 5) % 25]) & B[(i + 10) % 25]);
        }

        // Iota
        state[0] ^= keccak_round_constants[round];
    }

    // Squeeze phase - extract 32 bytes (256 bits)
    for (uint i = 0; i < 4; i++) {
        ulong val = state[i];
        for (uint j = 0; j < 8; j++) {
            output[i * 8 + j] = (uchar)(val >> (8 * j));
        }
    }
}