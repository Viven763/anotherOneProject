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

// Rotation offsets for all 25 positions (x + 5*y indexing)
__constant uint keccak_rho[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// Pi permutation: pi[i] gives the source index for destination i
// dest[i] = src[pi[i]]  where pi is the inverse of the standard pi mapping
__constant uint keccak_pi[25] = {
     0, 6, 12, 18, 24,
     3, 9, 10, 16, 22,
     1, 7, 13, 19, 20,
     4, 5, 11, 17, 23,
     2, 8, 14, 15, 21
};

// Keccak-f[1600] permutation
void keccak_f(ulong *state) {
    for (uint round = 0; round < KECCAK_ROUNDS; round++) {
        ulong C[5], D[5], B[25];

        // Theta step
        for (uint x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (uint x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (uint i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        // Rho and Pi steps combined
        for (uint i = 0; i < 25; i++) {
            uint src = keccak_pi[i];
            B[i] = rotl64(state[src], keccak_rho[src]);
        }

        // Chi step - operates within each row (same y)
        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                uint i = x + 5 * y;
                state[i] = B[i] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
            }
        }

        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

void keccak256(__generic const uchar *input, uint input_len, __generic uchar *output) {
    ulong state[25] = {0};
    uchar temp[136] = {0};  // Rate = 136 bytes for Keccak-256

    uint rate = 136; // 1088 bits / 8 = 136 bytes
    uint offset = 0;

    // Absorb phase - process full blocks
    while (input_len >= rate) {
        for (uint i = 0; i < rate / 8; i++) {
            ulong val = 0;
            for (uint j = 0; j < 8; j++) {
                val |= ((ulong)input[offset + i * 8 + j]) << (8 * j);
            }
            state[i] ^= val;
        }
        keccak_f(state);
        input_len -= rate;
        offset += rate;
    }

    // Padding - copy remaining input
    for (uint i = 0; i < 136; i++) {
        temp[i] = 0;
    }
    for (uint i = 0; i < input_len; i++) {
        temp[i] = input[offset + i];
    }
    
    // Keccak padding: pad with 0x01 ... 0x80
    temp[input_len] = 0x01;
    temp[rate - 1] |= 0x80;

    // Final block absorption
    for (uint i = 0; i < rate / 8; i++) {
        ulong val = 0;
        for (uint j = 0; j < 8; j++) {
            val |= ((ulong)temp[i * 8 + j]) << (8 * j);
        }
        state[i] ^= val;
    }

    // Final permutation
    keccak_f(state);

    // Squeeze phase - extract 32 bytes (256 bits)
    for (uint i = 0; i < 4; i++) {
        ulong val = state[i];
        for (uint j = 0; j < 8; j++) {
            output[i * 8 + j] = (uchar)(val >> (8 * j));
        }
    }
}
