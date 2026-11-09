import { describeExperiment } from '/state.mjs';
const icons = {
 home:'M3 11 12 3l9 8M5 10v11h5v-7h4v7h5V10',layers:'m3 7 9-4 9 4-9 5-9-5Zm0 5 9 5 9-5M3 17l9 5 9-5',database:'M21 6c0 2-4 4-9 4S3 8 3 6s4-4 9-4 9 2 9 4ZM3 6v12c0 2 4 4 9 4s9-2 9-4V6M3 12c0 2 4 4 9 4s9-2 9-4',chart:'M4 21V12h4v9M10 21V7h4v14M16 21V3h4v18',file:'M5 2h10l5 5v15H5V2Zm10 0v6h5M8 12h9M8 16h7',chip:'M6 6h12v12H6V6ZM3 8h3M3 12h3M3 16h3M18 8h3M18 12h3M18 16h3M8 3v3M12 3v3M16 3v3M8 18v3M12 18v3M16 18v3',trophy:'M7 3h10v7c0 7-10 7-10 0V3Zm0 2H3v3c0 4 4 5 5 5M17 5h4v3c0 4-4 5-5 5M12 16v6M7 22h10',image:'M3 4h18v16H3V4Zm1 13 5-6 4 4 3-4 5 6M15 8h.01',drop:'M12 2C10 7 4 10 4 15a8 8 0 0 0 16 0c0-5-6-8-8-13Z',grid:'M3 3h6v6H3V3Zm12 0h6v6h-6V3ZM3 15h6v6H3v-6Zm12 0h6v6h-6v-6Z'};
document.querySelectorAll('[data-icon]').forEach(el => { const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 24 24'); svg.setAttribute('fill','none'); svg.setAttribute('stroke','currentColor'); svg.setAttribute('stroke-width','1.8'); svg.setAttribute('stroke-linecap','round'); svg.setAttribute('stroke-linejoin','round'); svg.setAttribute('aria-hidden','true'); const p=document.createElementNS(svg.namespaceURI,'path'); p.setAttribute('d',icons[el.dataset.icon] || icons.file); svg.append(p); el.prepend(svg); });
const $ = id => document.getElementById(id);
const lung=document.createElementNS('http://www.w3.org/2000/svg','svg');lung.setAttribute('viewBox','0 0 40 40');lung.setAttribute('fill','none');lung.setAttribute('stroke','white');lung.setAttribute('stroke-width','2');const lungPath=document.createElementNS(lung.namespaceURI,'path');lungPath.setAttribute('d','M20 5v16m0-8-6 5m6-5 6 5M13 10C7 12 4 23 6 31c1 4 9 4 10 0V16m11-6c6 2 9 13 7 21-1 4-9 4-10 0V16');lung.append(lungPath);document.querySelector('.brand-icon').replaceChildren(lung);
let theme='light';try{theme=localStorage.getItem('cxr-theme')==='dark'?'dark':'light';}catch{}
function applyTheme(){document.documentElement.dataset.theme=theme;$('theme').textContent=theme==='dark'?'라이트 모드':'다크 모드';$('theme').setAttribute('aria-pressed',String(theme==='dark'));}
applyTheme();$('theme').addEventListener('click',()=>{theme=theme==='dark'?'light':'dark';applyTheme();try{localStorage.setItem('cxr-theme',theme);}catch{}});
let latest, busy = false;
let logTab='experiment';
for(const name of ['experiment','environment'])$(name+'-tab').addEventListener('click',()=>{logTab=name;for(const tab of ['experiment','environment'])$(tab+'-tab').setAttribute('aria-selected',String(tab===name));if(latest)render(latest);});
const bytes = n => n >= 1073741824 ? `${(n/1073741824).toFixed(2)} GB` : `${(n/1048576).toFixed(1)} MB`;
const metric = v => v === undefined || v === '' || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(4);
function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
async function action(name) {
 if (busy) return; busy = true; notice(''); $('start').disabled = true; $('stop').disabled = true;
 try { const r = await fetch(`/api/${name}`,{method:'POST'}); const data=await r.json(); if(!r.ok) throw Error(data.error); notice(name==='start'?'실험 시작을 요청했습니다. 실행 기록을 확인하고 있습니다.':'중단을 요청했습니다. 현재 작업을 안전하게 종료하고 있습니다.'); }
 catch(e){notice(e.message);} finally {busy=false;await refresh();}
}
$('start').addEventListener('click',()=>action('start')); $('stop').addEventListener('click',()=>action('stop'));
let paused=matchMedia('(prefers-reduced-motion: reduce)').matches;
function motion(){document.body.classList.toggle('paused',paused);$('motion').setAttribute('aria-pressed',String(paused));$('motion').firstChild.textContent=paused?'모션 재생 ':'모션 정지 ';}motion();$('motion').addEventListener('click',()=>{paused=!paused;motion();});
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelector('nav .selected')?.classList.remove('selected');a.classList.add('selected');}));
function render(s){
 latest=s;const run=s.run;const described=describeExperiment(s);$('status').textContent=described.status;
 $('start').disabled=busy||s.active||s.starting;$('stop').disabled=busy||!s.active||run?.status!=='running';
 $('gpu').textContent=s.gpu.name;$('gpu-detail').textContent=s.gpu.utilization===null?'실시간 장치 정보가 아직 없습니다.':`사용률 ${s.gpu.utilization}% · VRAM ${s.gpu.used} / ${s.gpu.total} MB`;
 $('result-count').textContent=s.results.length?`${s.results.length}개 평가 조건`:'아직 없음';
 $('download-summary').textContent=s.downloads.length?`${s.downloads.length}개 파일 · ${bytes(s.downloads.reduce((n,d)=>n+d.bytes,0))}`:'다운로드 파일 아직 없음';
 $('phase').textContent=run?.phase||'실험 시작 후 현재 작업이 표시됩니다.';
 const training=s.training;
 const phaseNames={verification:'검증 모델',retrieval_phase1:'검색 모델 출력부',retrieval_phase2:'검색 모델 전체',complete:'학습 완료',failed:'학습 오류',stopped:'학습 중단',preflight:'학습 입력 검사',data_validation:'학습 데이터 검증'};
 const statusNames={waiting_for_gpu:'GPU 여유 메모리 대기',training:'학습 중',initializing:'새 모델 초기화 중',validating:'검증 중',epoch_complete:'에폭 완료',completed:'완료',failed:'오류',stopped:'중단',ready:'입력 검사 완료'};
 $('training-state').textContent=training ? `${phaseNames[training.phase] || '학습 상태 확인'} · 에폭 ${training.epoch ?? '—'} / ${training.epochs ?? '—'}` : '데이터 검증 후 새 학습 시작';
 $('training-detail').textContent=training ? `단계 ${training.step ?? '—'} · 손실 ${metric(training.loss)} · ${statusNames[training.status] || '진행 상태 확인'}` : '공개 CXR 최종 가중치 사용 안 함 · 논문상 ImageNet 초기화';
 if(run?.error) notice(run.error);
 const log=s.log.text;const shown=logTab==='environment'?s.environment_log:s.log;$('terminal').textContent=shown.text||'[대기] 아직 실행 기록이 없습니다.';$('log-info').textContent=`${shown.truncated?'최근 64 KiB 범위 · 앞부분 생략 · ':''}원본 ${bytes(shown.bytes)}`;
 document.querySelectorAll('.stage').forEach((el,i)=>{
  const stage=described.stages[i];for(const cls of ['done','active','failed'])el.classList.toggle(cls,stage[cls]);el.querySelector('.stage-status').textContent=stage.label;
 });
 for(const name of ['original','blur','mosaic']){const image=$('preview-'+name);if(s.previews?.[name]&&!image.getAttribute('src')){image.onload=()=>{image.hidden=false;image.nextElementSibling.hidden=true;};image.onerror=()=>{image.hidden=true;image.nextElementSibling.hidden=false;image.removeAttribute('src');};image.src=`/api/preview/${name}.png`;}}
 // 준비 완료는 실제 실행 기록의 확인 메시지에 한해서 표시합니다.
 $('check-model').textContent=/모델.*(?:로딩 완료|로드 완료)|가중치.*완료/.test(log)?'✓':'○';
 $('check-data').textContent=s.downloads.length?'◒':'○';
 $('check-validation').textContent=/데이터 누수 검사.*중복 없음|데이터 검증.*완료/.test(log)?'✓':'○';
 $('downloads').replaceChildren(); if(!s.downloads.length)$('downloads').textContent='아직 다운로드 파일이 없습니다.';
 s.downloads.forEach(d=>{const line=document.createElement('div');line.textContent=`${d.partial?'다운로드 중':'보관 파일'} · ${d.name} · ${bytes(d.bytes)}`;$('downloads').append(line);});
 const tbody=$('result-rows');tbody.replaceChildren();s.results.forEach(r=>{const tr=document.createElement('tr');const stage=r.source.startsWith('stage_1')?'원본':r.source.startsWith('stage_2')?'블러':'모자이크';[stage,r.strength??'—',metric(r.auc),metric(stage==='원본'?r.precision_at_1:r.cmc_rank_1),metric(r.privacy_score),'기록됨'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});tbody.append(tr);});
 $('empty').hidden=!!s.results.length;$('updated').textContent=`최근 확인 ${new Date(s.updated_at).toLocaleTimeString('ko-KR',{timeZone:'Asia/Seoul'})} KST`;
}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('상태 조회 실패');render(await r.json());}catch(e){$('status').textContent='서버 연결 확인 필요';notice(e.message);$('start').disabled=true;$('stop').disabled=true;}}
refresh();setInterval(refresh,3000);
