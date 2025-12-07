// Database loader для Ethereum адресов
// Загружает БД и поддерживает быстрый binary search lookup

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

// Database record (4 bytes hash + 8 bytes address)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DbRecord {
    pub hash: u32,           // 4 байта (big-endian)
    pub addr_suffix: u64,    // 8 байт (little-endian)
}

// Implement OclPrm trait for GPU transfer
unsafe impl ocl::OclPrm for DbRecord {}

impl DbRecord {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 12, "Record must be 12 bytes");

        // Hash хранится в big-endian!
        let hash = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        // Addr suffix в little-endian
        let addr_suffix = u64::from_le_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
        ]);

        DbRecord { hash, addr_suffix }
    }

    pub fn is_empty(&self) -> bool {
        self.addr_suffix == 0
    }
}

// Database metadata structure
#[derive(Debug)]
pub struct DatabaseMetadata {
    pub db_length: u64,
    pub table_bytes: u64,
    pub bytes_per_addr: u32,
    pub null_addr: Vec<u8>,
    pub len: u64,
    pub max_len: u64,
    pub hash_bytes: u32,
    pub hash_mask: u32,
    pub last_filenum: Option<u32>,
    pub version: u32,
}

pub struct Database {
    pub metadata: DatabaseMetadata,
    pub records: Vec<DbRecord>,
}

impl Database {
    /// Load database from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and skip header line
        let mut header = String::new();
        reader.read_line(&mut header)?;
        println!("Database header: {}", header.trim());

        // Read metadata line
        let mut metadata_str = String::new();
        reader.read_line(&mut metadata_str)?;

        // Parse metadata
        let metadata = Self::parse_metadata(&metadata_str)?;
        println!("Database metadata: {:?}", metadata);

        // Read all records
        let mut records_data = Vec::new();
        reader.read_to_end(&mut records_data)?;

        // Convert to DbRecord structs
        let num_records = records_data.len() / 12;
        let mut records = Vec::with_capacity(num_records);

        for i in 0..num_records {
            let offset = i * 12;
            let record_bytes = &records_data[offset..offset + 12];
            records.push(DbRecord::from_bytes(record_bytes));
        }

        println!(
            "Loaded {} records ({} MB)",
            num_records,
            records_data.len() / 1_000_000
        );

        // Сортируем для быстрого binary search
        println!("Sorting records for binary search...");
        records.sort_by_key(|r| r.addr_suffix);
        println!("Sorted!");

        Ok(Database { metadata, records })
    }

    /// Parse Python dict-like metadata string
    fn parse_metadata(s: &str) -> Result<DatabaseMetadata, Box<dyn std::error::Error>> {
        let s = s.trim().trim_start_matches('{').trim_end_matches('}');

        let mut db_length = 0;
        let mut table_bytes = 0;
        let mut bytes_per_addr = 0;
        let null_addr = vec![0u8; 8];
        let mut len = 0;
        let mut max_len = 0;
        let mut hash_bytes = 0;
        let mut hash_mask = 0;
        let mut last_filenum = None;
        let mut version = 0;

        for pair in s.split(',') {
            let parts: Vec<&str> = pair.split(':').map(|s| s.trim()).collect();
            if parts.len() != 2 {
                continue;
            }

            let key = parts[0].trim_matches('\'').trim_matches('"');
            let value = parts[1];

            match key {
                "_dbLength" => db_length = value.parse()?,
                "_table_bytes" => table_bytes = value.parse()?,
                "_bytes_per_addr" => bytes_per_addr = value.parse()?,
                "_len" => len = value.parse()?,
                "_max_len" => max_len = value.parse()?,
                "_hash_bytes" => hash_bytes = value.parse()?,
                "_hash_mask" => hash_mask = value.parse()?,
                "version" => version = value.parse()?,
                "last_filenum" => {
                    if value != "None" {
                        last_filenum = Some(value.parse()?);
                    }
                }
                _ => {}
            }
        }

        Ok(DatabaseMetadata {
            db_length,
            table_bytes,
            bytes_per_addr,
            null_addr,
            len,
            max_len,
            hash_bytes,
            hash_mask,
            last_filenum,
            version,
        })
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.records.len() * std::mem::size_of::<DbRecord>()
    }

    /// Get statistics about the database
    pub fn stats(&self) -> DatabaseStats {
        let total_records = self.records.len();
        let empty_records = self.records.iter().filter(|r| r.is_empty()).count();
        let filled_records = total_records - empty_records;

        DatabaseStats {
            total_records,
            filled_records,
            empty_records,
            size_mb: self.size_bytes() / 1_000_000,
            load_factor: filled_records as f64 / total_records as f64,
        }
    }

    /// Быстрый lookup через binary search (БД отсортирована при загрузке)
    pub fn lookup_address_suffix(&self, addr_suffix: u64) -> bool {
        // Binary search: O(log n) вместо O(n)
        self.records.binary_search_by_key(&addr_suffix, |r| r.addr_suffix).is_ok()
    }

    /// Проверить полный ETH адрес (20 байт) в БД
    pub fn lookup_eth_address(&self, eth_address: &[u8; 20]) -> bool {
        // Берём последние 8 байт
        let addr_suffix = u64::from_le_bytes([
            eth_address[12], eth_address[13], eth_address[14], eth_address[15],
            eth_address[16], eth_address[17], eth_address[18], eth_address[19],
        ]);

        self.lookup_address_suffix(addr_suffix)
    }

    /// Получить raw данные для передачи в GPU
    pub fn get_raw_records(&self) -> &[DbRecord] {
        &self.records
    }

    /// Получить количество записей
    pub fn len(&self) -> usize {
        self.records.len()
    }
}

#[derive(Debug)]
pub struct DatabaseStats {
    pub total_records: usize,
    pub filled_records: usize,
    pub empty_records: usize,
    pub size_mb: usize,
    pub load_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_record_from_bytes() {
        let bytes = [
            0x01, 0x02, 0x03, 0x04, // hash (big-endian)
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, // addr_suffix
        ];

        let record = DbRecord::from_bytes(&bytes);
        assert_eq!(record.hash, 0x01020304); // big-endian
        assert_eq!(record.addr_suffix, 0x1817161514131211); // little-endian
    }

    #[test]
    fn test_empty_record() {
        let record = DbRecord {
            hash: 0,
            addr_suffix: 0,
        };
        assert!(record.is_empty());

        let record2 = DbRecord {
            hash: 123,
            addr_suffix: 456,
        };
        assert!(!record2.is_empty());
    }
}
