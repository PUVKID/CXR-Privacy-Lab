import { promises as fs } from 'node:fs';
const LIMIT = 65536;
export async function tail(file, limit = LIMIT) {
  let handle;
  try {
    handle = await fs.open(file, 'r');
    const { size } = await handle.stat();
    const start = Math.max(0, size - limit);
    const buffer = Buffer.alloc(Math.min(size, limit));
    await handle.read(buffer, 0, buffer.length, start);
    let bytes = buffer;
    if(start){const boundary=bytes.indexOf(10);bytes=boundary<0?bytes.subarray(bytes.length):bytes.subarray(boundary+1);}
    let text = new TextDecoder('utf-8',{fatal:true}).decode(bytes);
    // 중간 UTF-8 문자와 잘린 첫 행은 표시에서 제외합니다.

    return { text, bytes: size, truncated: start > 0 };
  } catch (e) { if (e.code === 'ENOENT') return { text: '', bytes: 0, truncated: false }; throw e; }
  finally { await handle?.close(); }
}
// 현재 실행의 고정 JSONL만 읽고 잘린 범위와 손상 기록을 별도로 알립니다.
export async function readTrainingHistory(file, runId, limit = 256 * 1024) {
  let handle;
  const result = { source: 'training/events.jsonl', bytes: 0, truncated: false, incomplete_tail: false, status: 'missing', last_observed_progress: null };
  try {
    handle = await fs.open(file, 'r');
    const { size } = await handle.stat(); result.bytes = size;
    const start = Math.max(0, size - limit); result.truncated = start > 0;
    const buffer = Buffer.alloc(Math.min(size, limit));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, start);
    let bytes = buffer.subarray(0, bytesRead);
    if (start) { const boundary = bytes.indexOf(10); bytes = boundary < 0 ? bytes.subarray(bytes.length) : bytes.subarray(boundary + 1); }
    result.incomplete_tail = bytes.length > 0 && bytes.at(-1) !== 10;
    if (result.incomplete_tail) { const boundary = bytes.lastIndexOf(10); bytes = boundary < 0 ? bytes.subarray(0, 0) : bytes.subarray(0, boundary + 1); }
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    const lines = text.split('\n'); lines.pop();
    let last = null;
    for (const line of lines) {
      const record = JSON.parse(line);
      if (!record || typeof record !== 'object' || Array.isArray(record)) throw new Error('invalid_record');
      if (Object.hasOwn(record, 'run_id') && record.run_id !== runId) throw new Error('different_run');
      if (record.status !== 'training') continue;
      const integer = key => Number.isSafeInteger(record[key]) && record[key] >= 0;
      if (!['verification', 'retrieval_phase1', 'retrieval_phase2'].includes(record.phase) ||
          !['epoch', 'epochs', 'step', 'global_step'].every(integer) ||
          typeof record.loss !== 'number' || !Number.isFinite(record.loss) ||
          typeof record.learning_rate !== 'number' || !Number.isFinite(record.learning_rate) || record.learning_rate < 0 ||
          typeof record.updated_at !== 'string' || !/^\d{4}-\d\d-\d\dT.*(?:Z|[+-]\d\d:\d\d)$/.test(record.updated_at) || !Number.isFinite(Date.parse(record.updated_at))) throw new Error('invalid_training_record');
      last = record;
    }
    result.status = result.incomplete_tail ? 'incomplete_tail' : result.truncated ? 'truncated' : 'ok';
    result.last_observed_progress = last;
    return result;
  } catch (error) {
    if (error.code === 'ENOENT') return result;
    // 중간의 잘못된 행은 건너뛰어 성공으로 처리하지 않고 관측값을 채택하지 않습니다.
    result.status = ['invalid_record', 'different_run', 'invalid_training_record'].includes(error.message) ? error.message : error instanceof SyntaxError || error.code === 'ERR_ENCODING_INVALID_ENCODED_DATA' ? 'malformed' : 'unreadable';
    return result;
  } finally { await handle?.close(); }
}
export function parseCSV(text) {
  const rows = []; let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (quoted && text[i + 1] === '"') { value += '"'; i++; } else quoted = !quoted; }
    else if (c === ',' && !quoted) { row.push(value); value = ''; }
    else if (c === '\n' && !quoted) { row.push(value.replace(/\r$/, '')); rows.push(row); row = []; value = ''; }
    else value += c;
  }
  if (value || row.length) { row.push(value.replace(/\r$/, '')); rows.push(row); }
  const fields = rows.shift()?.map(x => x.replace(/^\uFEFF/, '')) || [];
  return rows.filter(r => r.some(Boolean)).map(r => Object.fromEntries(fields.map((f, i) => [f, r[i] ?? ''])));
}
