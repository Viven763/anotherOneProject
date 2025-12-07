// Тест поиска адреса в БД (как делает Rust)
const fs = require('fs');

const DB_PATH = 'eth20240925';
const TARGET_SUFFIX = 'ba5fc6ee86e88bfd'; // last 8 bytes of target address

// Открываем файл
const fd = fs.openSync(DB_PATH, 'r');
const stats = fs.statSync(DB_PATH);

console.log('Database size:', stats.size, 'bytes');

// Читаем заголовок
const headerBuf = Buffer.alloc(500);
fs.readSync(fd, headerBuf, 0, 500, 0);

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

console.log('Header size:', headerEnd, 'bytes');

const dataSize = stats.size - headerEnd;
const numRecords = Math.floor(dataSize / 12);
const remainder = dataSize % 12;

console.log('Data size:', dataSize, 'bytes');
console.log('Number of 12-byte records:', numRecords);
console.log('Remainder bytes:', remainder);

// Ищем нашу запись среди последних 100 записей
console.log('\n=== Searching in last 100 records for target suffix ===');
console.log('Target suffix:', TARGET_SUFFIX);

const targetSuffixBuf = Buffer.from(TARGET_SUFFIX, 'hex');
let found = false;

for (let i = Math.max(0, numRecords - 100); i < numRecords; i++) {
    const recordBuf = Buffer.alloc(12);
    fs.readSync(fd, recordBuf, 0, 12, headerEnd + i * 12);
    const addrSuffix = recordBuf.slice(4, 12);
    
    if (addrSuffix.equals(targetSuffixBuf)) {
        console.log(`\n✅ FOUND at record ${i}!`);
        console.log('Record:', recordBuf.toString('hex'));
        found = true;
        break;
    }
}

if (!found) {
    console.log('\n❌ Not found in last 100 records');
    
    // Проверим последнюю запись отдельно
    console.log('\nLast record:');
    const lastRecordBuf = Buffer.alloc(12);
    fs.readSync(fd, lastRecordBuf, 0, 12, headerEnd + (numRecords - 1) * 12);
    console.log('Position:', headerEnd + (numRecords - 1) * 12);
    console.log('Raw:', lastRecordBuf.toString('hex'));
    console.log('Expected:', '2bf3767f' + TARGET_SUFFIX);
    
    // Также проверим что на самом деле в конце файла
    console.log('\nLast 24 bytes of file:');
    const endBuf = Buffer.alloc(24);
    fs.readSync(fd, endBuf, 0, 24, stats.size - 24);
    console.log('Raw:', endBuf.toString('hex'));
}

fs.closeSync(fd);

