// Создание тестовой базы данных с одним адресом
const fs = require('fs');

const DB_PATH = 'eth20240925';
const ADDRESS = '2bf3767f167a8a5da47ae1e5ba5fc6ee86e88bfd';

// Заголовок базы данных (как в оригинальной)
const header = 'eth_addresses_db\n';
const metadata = "{'_dbLength': 1, '_table_bytes': 12, '_bytes_per_addr': 8, '_len': 1, '_max_len': 1, '_hash_bytes': 4, '_hash_mask': 0, 'version': 1, 'last_filenum': None}\n";

// Парсим адрес
const addrBytes = Buffer.from(ADDRESS, 'hex');
console.log('Address:', '0x' + ADDRESS);
console.log('Address bytes:', addrBytes);

// Последние 8 байт адреса (позиции 12-19)
const addrSuffix = addrBytes.slice(12, 20);
console.log('Address suffix (last 8 bytes):', addrSuffix.toString('hex'));

// Первые 4 байта как hash
const hashBytes = addrBytes.slice(0, 4);
console.log('Hash bytes (first 4):', hashBytes.toString('hex'));

// Создаём запись (12 байт)
const record = Buffer.alloc(12);
// Hash в big-endian
record[0] = hashBytes[0];
record[1] = hashBytes[1];
record[2] = hashBytes[2];
record[3] = hashBytes[3];
// Addr suffix как есть (уже в little-endian порядке в Buffer)
addrSuffix.copy(record, 4);

console.log('Record (12 bytes):', record.toString('hex'));

// Собираем файл
const headerBuf = Buffer.from(header, 'utf8');
const metadataBuf = Buffer.from(metadata, 'utf8');
const dbFile = Buffer.concat([headerBuf, metadataBuf, record]);

fs.writeFileSync(DB_PATH, dbFile);

console.log('\n✅ База данных создана!');
console.log('File:', DB_PATH);
console.log('Size:', dbFile.length, 'bytes');
console.log('Header size:', headerBuf.length + metadataBuf.length, 'bytes');
console.log('Records:', 1);

