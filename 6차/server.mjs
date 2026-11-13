import http from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFile } from 'node:child_process';
import { EXPERIMENT_PLAN } from './화면/state.mjs';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const LIMIT = 65536;
export async function tail(file, limit = LIMIT) {
  let handle;
  try {
    handle = await fs.open(file, 'r');
    const { size } = await handle.stat();
    const start = Math.max(0, size - limit);
    const buffer = Buffer.alloc(Math.min(size, limit));
    await handle.read(buffer, 0, buffer.length, start);
    let text = buffer.toString('utf8');
    // 중간 UTF-8 문자와 잘린 첫 행은 표시에서 제외합니다.
    if (start) text = text.slice(text.indexOf('\n') + 1);
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
async function json(file) { try { return JSON.parse(await fs.readFile(file, 'utf8')); } catch (e) { if (e.code === 'ENOENT' || e instanceof SyntaxError) return null; throw e; } }
function alive(pid) { if (!Number.isInteger(pid) || pid < 1) return false; try { process.kill(pid, 0); return true; } catch { return false; } }
export function createDashboard(options = {}) {
  const root = options.root || ROOT;
  const executionRoot = options.executionRoot || (options.root ? root : path.resolve(ROOT, '../5차'));
  const research = options.research || path.resolve(executionRoot, '../연구 자료');
  const control = options.control || (options.research ? path.join(research, 'control') : path.join(executionRoot, 'work', 'control'));
  const previewControl = options.research ? control : path.join(research, 'control');
  const launch = options.launch || (() => spawn(path.resolve(executionRoot, '../실행 환경/Scripts/python.exe'), ['-B', '-X', 'utf8', path.join(executionRoot, 'run_experiment.py')], { cwd: executionRoot, windowsHide: true, stdio: 'ignore' }));
  let pending = false, localChild = null, gpuCache = { name: '확인 중', utilization: null }, gpuAt = 0;
  async function state() {
    const run = await json(path.join(control, 'run.json'));
    const active = run?.status === 'running' && alive(run.pid);
    if (run?.status === 'running' && !active) { run.status = 'interrupted'; run.error = '실행 프로세스가 확인되지 않습니다.'; }
    return { run, active: active || !!localChild, starting: pending };
  }
  async function status() {
    const current = await state();
    const results = [];
    // 현재 실행의 결과만 읽으며 과거 가중치 평가와 새 학습 결과를 섞지 않습니다.
    const runId=current.run?.run_id;
    const validRun=typeof runId==='string' && /^[a-zA-Z0-9_-]{1,80}$/.test(runId);
    const resultRoot=validRun ? path.join(research,'results','runs',runId) : (options.research ? path.join(research,'results') : null);
    for (const folder of ['stage_1_original', 'stage_2_blur', 'stage_3_mosaic']) {
      try {
        if (!resultRoot) continue;
        const file = path.join(resultRoot, folder, 'summary.csv');
        const stat = await fs.stat(file);
        if (stat.size <= 2 * 1024 * 1024) for (const row of parseCSV(await fs.readFile(file, 'utf8'))) results.push({ ...row, source: folder });
      } catch (e) { if (e.code !== 'ENOENT') throw e; }
    }
    const downloads = [];
    try { for (const name of (await fs.readdir(path.join(research, 'downloads'))).slice(0, 200)) {
      const stat = await fs.stat(path.join(research, 'downloads', name));
      if (stat.isFile()) downloads.push({ name, bytes: stat.size, partial: name.endsWith('.part') });
    } } catch (e) { if (e.code !== 'ENOENT') throw e; }
    if (Date.now() - gpuAt > 15000) {
      gpuAt = Date.now();
      (options.gpuQuery || execFile)('nvidia-smi', ['--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], { windowsHide: true, timeout: 2000, maxBuffer: 4096 }, (err, out) => {
        if (err) gpuCache = { name: '조회 불가', utilization: null };
        else { const [name, utilization, used, total] = out.trim().split('\n')[0].split(',').map(s => s.trim()); gpuCache = { name, utilization, used, total }; }
      });
    }
    const previews={};
    for (const name of ['original','blur','mosaic']) {
      try { const info=await fs.stat(path.join(previewControl,'previews',`${name}.png`)); previews[name]=info.isFile()&&info.size>0; }
      catch(e) { if(e.code!=='ENOENT')throw e; previews[name]=false; }
    }
    const trainingRoot=validRun?path.join(executionRoot,'work','runs',runId,'training'):null;
    const training=trainingRoot?await json(path.join(trainingRoot,'progress.json')):null;
    const training_history=trainingRoot?await readTrainingHistory(path.join(trainingRoot,'events.jsonl'),runId):null;
    const last_observed_progress=training_history?.last_observed_progress??null;
    const training_phases={};
    for(const phase of ['verification','retrieval_phase1','retrieval_phase2']){
      let selected_weights=false;
      if(trainingRoot){try{selected_weights=(await fs.stat(path.join(trainingRoot,phase,'selected_weights.pth'))).isFile();}catch(e){if(e.code!=='ENOENT')throw e;}}
      training_phases[phase]={initialization:trainingRoot?await json(path.join(trainingRoot,phase,'initialization.json')):null,first_backward:trainingRoot?await json(path.join(trainingRoot,phase,'first_backward.json')):null,selected_weights};
    }
    const artifacts=trainingRoot?await json(path.join(trainingRoot,'artifacts.json')):null;
    const preflight=trainingRoot?await json(path.join(trainingRoot,'preflight.json')):null;
    const experiment={screen_revision:6,execution_revision:current.run?.revision??5,condition_count:21,shared_models:true,plan:EXPERIMENT_PLAN,denominators:preflight?{verification_test_images:preflight.verification_counts?.test,retrieval_test_images:preflight.retrieval_counts?.test,verification_test_pairs:preflight.verification_official_projected_pair_counts?.test}:null};
    return { ...current, revision:6, experiment, training,last_observed_progress,training_history,training_phases,training_artifacts_complete:!!(artifacts?.verification_checkpoint&&artifacts?.retrieval_checkpoint), gpu: gpuCache, results, downloads, previews, log: await tail(path.join(control, 'experiment.log')), environment_log: await tail(path.join(research,'control','environment.log')), environment_ready: !!(await json(path.join(research,'control','environment_ready.json'))), updated_at: new Date().toISOString() };
  }
  const server = http.createServer(async (req, res) => {
    const send = (code, data) => { res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(data)); };
    try {
      const host = `127.0.0.1:${server.address().port}`;
      if (req.headers.host !== host) return send(403, { error: '로컬 Host만 허용합니다.' });
      if (req.method === 'GET' && req.url === '/api/status') return send(200, await status());
      const previewFiles={'/api/preview/original.png':'original.png','/api/preview/blur.png':'blur.png','/api/preview/mosaic.png':'mosaic.png'};
      if (req.method==='GET' && Object.hasOwn(previewFiles,req.url)) {
        try {
          const file=path.join(previewControl,'previews',previewFiles[req.url]);const info=await fs.stat(file);
          if(!info.isFile() || info.size>10*1024*1024)return send(404,{error:'미리보기를 준비하고 있습니다.'});
          const data=await fs.readFile(file);
          res.writeHead(200,{'Content-Type':'image/png','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});return res.end(data);
        } catch(e) { if(e.code==='ENOENT')return send(404,{error:'미리보기를 준비하고 있습니다.'});throw e; }
      }
      if (req.method === 'POST') {
        if (req.headers.origin !== `http://${host}`) return send(403, { error: '로컬 Origin만 허용합니다.' });
        if (Number(req.headers['content-length'] || 0) > 0 || req.headers['transfer-encoding']) return send(400, { error: '요청 본문은 허용하지 않습니다.' });
        if (req.url === '/api/start') {
          if (pending) return send(409, { error: '실험을 시작하고 있습니다.' });
          pending = true;
          try {
            if ((await state()).active) return send(409, { error: '이미 실행 중입니다.' });
            const child = launch(); localChild = child;
            const clear = () => { if (localChild === child) localChild = null; };
            child.once('exit', clear); child.once('error', clear);
            await new Promise((resolve, reject) => { child.once('spawn', resolve); child.once('error', reject); });
            return send(202, { status: 'starting' });
          } finally { pending = false; }
        }
        if (req.url === '/api/stop') {
          const { run, active } = await state();
          if (!active || !run?.run_id || run.status !== 'running') return send(409, { error: '실행 상태가 기록된 후 중단할 수 있습니다.' });
          await fs.mkdir(control, { recursive: true });
          await fs.writeFile(path.join(control, 'stop.request'), run.run_id, 'utf8');
          return send(202, { status: 'stop_requested' });
        }
        return send(404, { error: '없는 요청입니다.' });
      }
      if (req.method !== 'GET') return send(405, { error: '허용하지 않는 방식입니다.' });
      const files = { '/': ['index.html', 'text/html'], '/style.css': ['style.css', 'text/css'], '/app.js': ['app.js', 'text/javascript'], '/state.mjs': ['state.mjs','text/javascript'], '/assets/NanumGothic-Bold.ttf': ['assets/NanumGothic-Bold.ttf', 'font/ttf'], '/assets/OFL.txt': ['assets/OFL.txt', 'text/plain'] };
      const entry = files[req.url]; if (!entry) return send(404, { error: '파일이 없습니다.' });
      const data = await fs.readFile(path.join(root, '화면', entry[0]));
      res.writeHead(200, { 'Content-Type': `${entry[1]}; charset=utf-8`, 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'", 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer' }); res.end(data);
    } catch (e) { send(500, { error: e.code === 'ENOENT' ? '필요한 실행 파일이 없습니다.' : '로컬 요청 처리에 실패했습니다.' }); }
  });
  return server;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const server=createDashboard();
  server.on('error', error => { console.error(`[대시보드 시작 실패] ${error.code}: ${error.message}`); process.exitCode=1; });
  server.listen(8787, '127.0.0.1', () => console.log('화면 6차 · 실험 5차: http://127.0.0.1:8787'));
}
