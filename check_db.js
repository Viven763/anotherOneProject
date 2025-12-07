// Проверка формата записей в базе данных
const fs = require('fs');

const DB_PATH = 'eth20240925';

// Открываем файл
const fd = fs.openSync(DB_PATH, 'r');
const stats = fs.statSync(DB_PATH);

console.log('Database size:', (stats.size / 1024 / 1024 / 1024).toFixed(2), 'GB');

// Читаем заголовок (первые 500 байт)
const headerBuf = Buffer.alloc(500);
fs.readSync(fd, headerBuf, 0, 500, 0);

// Находим конец заголовка (2 строки с \n)
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

console.log('\n=== HEADER ===');
console.log(headerBuf.slice(0, headerEnd).toString());
console.log('Header ends at byte:', headerEnd);

// Читаем первые 5 записей
console.log('\n=== FIRST 5 RECORDS ===');
const recordBuf = Buffer.alloc(12);
for (let i = 0; i < 5; i++) {
    fs.readSync(fd, recordBuf, 0, 12, headerEnd + i * 12);
    const hash = recordBuf.slice(0, 4).toString('hex');
    const addrSuffix = recordBuf.slice(4, 12).toString('hex');
    console.log(`Record ${i}: hash=${hash}, addr_suffix=${addrSuffix}, raw=${recordBuf.toString('hex')}`);
}

// Читаем последние 5 записей (включая только что добавленную)
console.log('\n=== LAST 5 RECORDS ===');
const numRecords = Math.floor((stats.size - headerEnd) / 12);
console.log('Total records:', numRecords);

for (let i = Math.max(0, numRecords - 5); i < numRecords; i++) {
    fs.readSync(fd, recordBuf, 0, 12, headerEnd + i * 12);
    const hash = recordBuf.slice(0, 4).toString('hex');
    const addrSuffix = recordBuf.slice(4, 12).toString('hex');
    console.log(`Record ${i}: hash=${hash}, addr_suffix=${addrSuffix}, raw=${recordBuf.toString('hex')}`);
}

// Проверяем нашу запись
console.log('\n=== OUR TARGET ADDRESS ===');
console.log('Target: 0x2bf3767f167a8a5da47ae1e5ba5fc6ee86e88bfd');
console.log('Expected hash (first 4 bytes): 2bf3767f');
console.log('Expected suffix (last 8 bytes): ba5fc6ee86e88bfd');

fs.closeSync(fd);

