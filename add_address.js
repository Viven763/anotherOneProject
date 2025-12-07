// Скрипт для добавления ETH адреса в базу данных (append mode)
const fs = require('fs');

const DB_PATH = 'eth20240925';
const NEW_ADDRESS = '2bf3767f167a8a5da47ae1e5ba5fc6ee86e88bfd';

// Получаем информацию о файле
const stats = fs.statSync(DB_PATH);
console.log('Database size:', (stats.size / 1024 / 1024 / 1024).toFixed(2), 'GB');

// Парсим адрес
const addrBytes = Buffer.from(NEW_ADDRESS, 'hex');
console.log('Address:', '0x' + NEW_ADDRESS);

// Последние 8 байт адреса (позиции 12-19)
const addrSuffix = addrBytes.slice(12, 20);
console.log('Address suffix (last 8 bytes):', addrSuffix.toString('hex'));

// Первые 4 байта как hash
const hashBytes = addrBytes.slice(0, 4);
console.log('Hash bytes (first 4):', hashBytes.toString('hex'));

// Создаём новую запись (12 байт)
const newRecord = Buffer.alloc(12);
// Hash в big-endian
newRecord[0] = hashBytes[0];
newRecord[1] = hashBytes[1];
newRecord[2] = hashBytes[2];
newRecord[3] = hashBytes[3];
// Addr suffix копируем как есть
addrSuffix.copy(newRecord, 4);

console.log('New record:', newRecord.toString('hex'));

// Дописываем запись в конец файла
fs.appendFileSync(DB_PATH, newRecord);

const newStats = fs.statSync(DB_PATH);
console.log('\n✅ Адрес добавлен в базу данных!');
console.log('Address:', '0x' + NEW_ADDRESS);
console.log('Database size:', stats.size, '->', newStats.size, 'bytes');
console.log('Added:', newStats.size - stats.size, 'bytes');
