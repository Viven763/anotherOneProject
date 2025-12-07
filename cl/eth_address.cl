// Ethereum address derivation from public key
// Ethereum address = last 20 bytes of keccak256(uncompressed_public_key[1:])

// Derive Ethereum address from extended public key
void eth_address_for_public_key(extended_public_key_t *pub, __private uchar *eth_address) {
    // Get uncompressed public key (65 bytes: 0x04 + 32 bytes X + 32 bytes Y)
    uchar uncompressed_pubkey[65] = {0};
    uncompressed_public_key(pub, uncompressed_pubkey);

    // Ethereum uses keccak256(pubkey[1:]) - skip the 0x04 prefix
    uchar keccak_hash[32] = {0};
    keccak256(uncompressed_pubkey + 1, 64, keccak_hash);

    // Ethereum address is the last 20 bytes of the keccak256 hash
    for(int i = 0; i < 20; i++) {
        eth_address[i] = keccak_hash[i + 12];
    }
}

// Derive Ethereum address from BIP44 path: m/44'/60'/0'/0/0
// This is the standard Ethereum derivation path
void derive_eth_address_bip44(__private uchar *seed, __private uchar *eth_address) {
    // Create master key from seed
    extended_private_key_t master;
    new_master_from_seed(BITCOIN_MAINNET, seed, &master); // Network doesn't matter for Ethereum

    // BIP44 path for Ethereum: m/44'/60'/0'/0/0
    // m/44' (purpose)
    extended_private_key_t purpose;
    hardened_private_child_from_private(&master, &purpose, 44);

    // m/44'/60' (coin_type - 60 is Ethereum)
    extended_private_key_t coin_type;
    hardened_private_child_from_private(&purpose, &coin_type, 60);

    // m/44'/60'/0' (account)
    extended_private_key_t account;
    hardened_private_child_from_private(&coin_type, &account, 0);

    // m/44'/60'/0'/0 (change - 0 for external)
    extended_private_key_t change;
    normal_private_child_from_private(&account, &change, 0);

    // m/44'/60'/0'/0/0 (address_index)
    extended_private_key_t address_key;
    normal_private_child_from_private(&change, &address_key, 0);

    // Convert to public key
    extended_public_key_t pub;
    public_from_private(&address_key, &pub);

    // Derive Ethereum address
    eth_address_for_public_key(&pub, eth_address);
}

// Alternative: Derive multiple Ethereum addresses (for checking multiple derivation paths)
void derive_eth_addresses_multiple_paths(__private uchar *seed, __private uchar *addresses, int num_addresses) {
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

    // Derive multiple addresses (0 to num_addresses-1)
    for(int i = 0; i < num_addresses; i++) {
        extended_private_key_t address_key;
        normal_private_child_from_private(&change, &address_key, i);

        extended_public_key_t pub;
        public_from_private(&address_key, &pub);

        eth_address_for_public_key(&pub, &addresses[i * 20]);
    }
}
