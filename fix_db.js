// Фикс: удаляем неправильно добавленную запись и добавляем правильно
const fs = require('fs');

const DB_PATH = 'eth20240925';
const NEW_ADDRESS = '2bf3767f167a8a5da47ae1e5ba5fc6ee86e88bfd';

const stats = fs.statSync(DB_PATH);
console.log('Current size:', stats.size, 'bytes');

// Удаляем последние 12 байт (неправильно добавленная запись)
fs.truncateSync(DB_PATH, stats.size - 12);
console.log('Truncated to:', stats.size - 12, 'bytes');

// Проверяем выравнивание
const newStats = fs.statSync(DB_PATH);
const fd = fs.openSync(DB_PATH, 'r');
const headerBuf = Buffer.alloc(300);
fs.readSync(fd, headerBuf, 0, 300, 0);

let headerEnd = 0;
let newlineCount = 0;
for (let i = 0; i < headerBuf.length; i++) {
    if (headerBuf[i] === 0x0a) {
        newlineCount++;
        if (newlineCount === 2) {
            headerEnd = i + 1;
            break;
        }
    }
}
fs.closeSync(fd);

const dataSize = newStats.size - headerEnd;
const remainder = dataSize % 12;
console.log('Header size:', headerEnd);
console.log('Data size:', dataSize);
console.log('Remainder:', remainder, '(need', 12 - remainder, 'bytes padding)');

// Добавляем padding для выравнивания
const paddingNeeded = (12 - remainder) % 12;
if (paddingNeeded > 0) {
    const padding = Buffer.alloc(paddingNeeded, 0);
    fs.appendFileSync(DB_PATH, padding);
    console.log('Added', paddingNeeded, 'bytes padding');
}

// Теперь добавляем нашу запись правильно
const addrBytes = Buffer.from(NEW_ADDRESS, 'hex');
const newRecord = Buffer.alloc(12);
// Hash в big-endian (первые 4 байта адреса)
newRecord[0] = addrBytes[0];
newRecord[1] = addrBytes[1];
newRecord[2] = addrBytes[2];
newRecord[3] = addrBytes[3];
// Addr suffix (последние 8 байт)
addrBytes.slice(12, 20).copy(newRecord, 4);

console.log('New record:', newRecord.toString('hex'));
fs.appendFileSync(DB_PATH, newRecord);

// Проверяем результат
const finalStats = fs.statSync(DB_PATH);
const finalDataSize = finalStats.size - headerEnd;
const finalRemainder = finalDataSize % 12;
const finalRecords = Math.floor(finalDataSize / 12);

console.log('\n✅ Done!');
console.log('Final size:', finalStats.size, 'bytes');
console.log('Final records:', finalRecords);
console.log('Final remainder:', finalRemainder, '(should be 0)');

