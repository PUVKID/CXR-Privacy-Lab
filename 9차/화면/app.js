import { describeExperiment, formatPercent, formatRaw, percentNotice, EXPERIMENT_PLAN, CONDITIONS, DATASETS, TASKS, conditionState, trainingPhaseStates, trainingDisplay, historyNotice, trainingFractions, lossSegments, phaseTimeline, formatProgressPercent } from '/state.mjs';
const icons = {
 home:'M3 11 12 3l9 8M5 10v11h5v-7h4v7h5V10',layers:'m3 7 9-4 9 4-9 5-9-5Zm0 5 9 5 9-5M3 17l9 5 9-5',database:'M21 6c0 2-4 4-9 4S3 8 3 6s4-4 9-4 9 2 9 4ZM3 6v12c0 2 4 4 9 4s9-2 9-4V6M3 12c0 2 4 4 9 4s9-2 9-4',chart:'M4 21V12h4v9M10 21V7h4v14M16 21V3h4v18',file:'M5 2h10l5 5v15H5V2Zm10 0v6h5M8 12h9M8 16h7',chip:'M6 6h12v12H6V6ZM3 8h3M3 12h3M3 16h3M18 8h3M18 12h3M18 16h3M8 3v3M12 3v3M16 3v3M8 18v3M12 18v3M16 18v3',trophy:'M7 3h10v7c0 7-10 7-10 0V3Zm0 2H3v3c0 4 4 5 5 5M17 5h4v3c0 4-4 5-5 5M12 16v6M7 22h10',image:'M3 4h18v16H3V4Zm1 13 5-6 4 4 3-4 5 6M15 8h.01',drop:'M12 2C10 7 4 10 4 15a8 8 0 0 0 16 0c0-5-6-8-8-13Z',grid:'M3 3h6v6H3V3Zm12 0h6v6h-6V3ZM3 15h6v6H3v-6Zm12 0h6v6h-6v-6Z'};
document.querySelectorAll('[data-icon]').forEach(el => { const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 24 24'); svg.setAttribute('fill','none'); svg.setAttribute('stroke','currentColor'); svg.setAttribute('stroke-width','1.8'); svg.setAttribute('stroke-linecap','round'); svg.setAttribute('stroke-linejoin','round'); svg.setAttribute('aria-hidden','true'); const p=document.createElementNS(svg.namespaceURI,'path'); p.setAttribute('d',icons[el.dataset.icon] || icons.file); svg.append(p); el.prepend(svg); });
const $ = id => document.getElementById(id);
const lung=document.createElementNS('http://www.w3.org/2000/svg','svg');lung.setAttribute('viewBox','0 0 40 40');lung.setAttribute('fill','none');lung.setAttribute('stroke','white');lung.setAttribute('stroke-width','2');const lungPath=document.createElementNS(lung.namespaceURI,'path');lungPath.setAttribute('d','M20 5v16m0-8-6 5m6-5 6 5M13 10C7 12 4 23 6 31c1 4 9 4 10 0V16m11-6c6 2 9 13 7 21-1 4-9 4-10 0V16');lung.append(lungPath);document.querySelector('.brand-icon').replaceChildren(lung);
let theme='light';try{theme=localStorage.getItem('cxr-theme')==='dark'?'dark':'light';}catch{}
function applyTheme(){document.documentElement.dataset.theme=theme;$('theme').textContent=theme==='dark'?'라이트 모드':'다크 모드';$('theme').setAttribute('aria-pressed',String(theme==='dark'));}
applyTheme();$('theme').addEventListener('click',()=>{theme=theme==='dark'?'light':'dark';applyTheme();try{localStorage.setItem('cxr-theme',theme);}catch{}});
let latest,logTab='training';
const metric=formatRaw;
const observedTime=value=>Number.isFinite(Date.parse(value))?new Date(value).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'})+' KST':'시각 확인 불가';
function notice(text){$('notice').textContent=text;$('notice').hidden=!text;}
for(const [id,name] of [['experiment-tab','training'],['environment-tab','privacy']])$(id).addEventListener('click',()=>{logTab=name;$(id).setAttribute('aria-selected','true');$(id==='experiment-tab'?'environment-tab':'experiment-tab').setAttribute('aria-selected','false');if(latest)renderLog(latest);});
let paused=matchMedia('(prefers-reduced-motion: reduce)').matches;
function motion(){document.body.classList.toggle('paused',paused);$('motion').setAttribute('aria-pressed',String(paused));$('motion').firstChild.textContent=paused?'모션 재생 ':'모션 정지 ';}motion();$('motion').addEventListener('click',()=>{paused=!paused;motion();});
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelector('nav .selected')?.classList.remove('selected');a.classList.add('selected');}));

const selectedStrengths={1:0,2:0,3:0};let selectedStage=1,dataset='NIH',task='verification';
function readSelection(){
 const match=/^#dataset-(NIH|COVID19)-task-(verification|retrieval)-stage-([123])-strength-(\d+)$/.exec(location.hash);
 if(match){const stage=Number(match[3]),strength=Number(match[4]);if(EXPERIMENT_PLAN[stage-1].strengths.includes(strength)){dataset=match[1];task=match[2];selectedStage=stage;selectedStrengths[stage]=strength;}}
}
readSelection();
function selectCondition(stage,strength=selectedStrengths[stage]){
 if(!EXPERIMENT_PLAN[stage-1]?.strengths.includes(strength))return;
 selectedStage=stage;selectedStrengths[stage]=strength;
 history.replaceState(null,'',`#dataset-${dataset}-task-${task}-stage-${stage}-strength-${strength}`);
 if(latest)renderExplorer(latest);
}
window.addEventListener('hashchange',()=>{readSelection();if(latest)renderExplorer(latest);});
for(const button of document.querySelectorAll('[data-dataset]'))button.addEventListener('click',()=>{dataset=button.dataset.dataset;selectCondition(selectedStage);});
for(const button of document.querySelectorAll('[data-task]'))button.addEventListener('click',()=>{task=button.dataset.task;selectCondition(selectedStage);});
for(const plan of EXPERIMENT_PLAN){
 const tab=document.createElement('button');tab.type='button';tab.id=`condition-tab-${plan.stage}`;tab.setAttribute('role','tab');tab.setAttribute('aria-controls','condition-detail');tab.dataset.stageSelect=String(plan.stage);tab.textContent=`${plan.title} · ${plan.strengths.length}조건`;tab.addEventListener('click',()=>selectCondition(plan.stage));
 tab.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const index=event.key==='Home'?0:event.key==='End'?2:(plan.stage-1+(event.key==='ArrowRight'?1:2))%3;selectCondition(index+1);$(`condition-tab-${index+1}`).focus();}});$('condition-tabs').append(tab);
 const group=document.createElement('section');group.className='condition-group';group.dataset.stage=String(plan.stage);group.setAttribute('aria-label',plan.title+' 강도');
 const title=document.createElement('h3');title.textContent=plan.title+(plan.stage>1?' · 확장 비교':' · 기준');const chips=document.createElement('div');chips.className='chips';
 for(const [index,strength] of plan.strengths.entries()){
  const chip=document.createElement('button');chip.type='button';chip.className='condition-chip';chip.dataset.stage=String(plan.stage);chip.dataset.strength=String(strength);chip.id=`condition-${plan.stage}-${strength}`;
  const label=document.createElement('span'),status=document.createElement('small');label.textContent=plan.stage===1?'원본 0':String(strength)+' px';status.textContent='대기';chip.append(label,status);
  chip.addEventListener('click',()=>selectCondition(plan.stage,strength));chip.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?plan.strengths.length-1:(index+(event.key==='ArrowRight'?1:plan.strengths.length-1))%plan.strengths.length;selectCondition(plan.stage,plan.strengths[next]);$(`condition-${plan.stage}-${plan.strengths[next]}`).focus();}});chips.append(chip);
 }
 group.append(title,chips);$('condition-chips').append(group);
}
document.querySelectorAll('.stage-pick').forEach(button=>button.addEventListener('click',()=>{selectCondition(Number(button.dataset.stageSelect));$('condition-explorer').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});}));
function renderExplorer(s){
 const selected=CONDITIONS.find(c=>c.stage===selectedStage&&c.strength===selectedStrengths[selectedStage]);
 for(const button of document.querySelectorAll('[data-dataset]'))button.setAttribute('aria-pressed',String(button.dataset.dataset===dataset));
 for(const button of document.querySelectorAll('[data-task]'))button.setAttribute('aria-pressed',String(button.dataset.task===task));
 for(const plan of EXPERIMENT_PLAN){
  const tab=$(`condition-tab-${plan.stage}`),chosen=plan.stage===selectedStage;tab.setAttribute('aria-selected',String(chosen));tab.tabIndex=chosen?0:-1;
  document.querySelector(`.condition-group[data-stage="${plan.stage}"]`).classList.toggle('selected-group',chosen);
  document.querySelector(`.stage-pick[data-stage-select="${plan.stage}"]`).setAttribute('aria-pressed',String(chosen));
  for(const strength of plan.strengths){
   const item=CONDITIONS.find(c=>c.stage===plan.stage&&c.strength===strength),state=conditionState(s,dataset,task,item.id),chip=$(`condition-${plan.stage}-${strength}`);
   chip.setAttribute('aria-pressed',String(chosen&&strength===selected.strength));chip.dataset.status=state.status;
   chip.querySelector('small').textContent={complete:'결과 있음',waiting:'미측정',error:'확인 필요',observed:'최근 기록'}[state.status];
   chip.title=`${dataset} · ${task==='verification'?'검증':'검색'} · ${plan.title} ${strength} px · ${state.label}`;
  }
 }
 const state=conditionState(s,dataset,task,selected.id),row=state.row,values=row?.metrics;
 $('condition-detail').setAttribute('aria-labelledby',`condition-tab-${selectedStage}`);
 $('condition-title').textContent=`${dataset==='NIH'?'NIH':'COVID-19'} · ${task==='verification'?'환자 동일 여부 검증':'같은 환자 영상 검색'} · ${EXPERIMENT_PLAN[selectedStage-1].title} · ${selected.strength}${selectedStage>1?' px · 확장 비교':''}`;
 $('condition-status').textContent=state.label;
 $('condition-description').textContent=selected.mode==='original'?'변형하지 않은 원본으로 기준 성능을 측정합니다.':selected.mode==='blur'?'원영상에서 GaussianBlur 반경을 적용합니다. 강도 단위는 모델 입력 전 원영상의 픽셀입니다.':'원영상에서 타일 크기 기준으로 BOX 축소 후 NEAREST 확대합니다. 단위는 원영상 픽셀입니다.';
 $('condition-model').textContent='검증 모델과 검색 모델을 한 번 학습한 뒤 동일 모델을 21조건에서 공유합니다. 강도별로 재학습하지 않습니다.';
 $('condition-input').textContent=task==='verification'?'변환한 왼쪽 영상과 원본 오른쪽 영상을 비교합니다.':'변환한 질의와 원본 갤러리를 비교하며 같은 촬영 이미지 자체는 제외합니다.';
 const labels=task==='verification'?['AUC · 원값','검증 정확도','검증 쌍 수','신뢰구간 · 원기록']:['검색 P@1','mAP@R','R-Precision','프라이버시 점수'];
 ['metric-first-label','metric-second-label','metric-third-label','metric-fourth-label'].forEach((id,i)=>$(id).textContent=labels[i]);
 const output=task==='verification'?[metric(values?.auc),formatPercent(values?.accuracy),metric(values?.n_pairs),values?.confidence_interval?JSON.stringify(values.confidence_interval):'미측정']:[formatPercent(values?.precision_at_1),formatPercent(values?.map_at_r),formatPercent(values?.r_precision),formatPercent(values?.privacy_score_at_1,{ratio:false})];
 ['condition-auc','condition-rank','condition-privacy','condition-fourth'].forEach((id,i)=>$(id).textContent=output[i]);
 const percentages=task==='verification'?[[values?.accuracy,{}]]:[[values?.precision_at_1,{}],[values?.map_at_r,{}],[values?.r_precision,{}],[values?.privacy_score_at_1,{ratio:false}]];
 $('condition-rounding').textContent=percentages.map(([value,options])=>percentNotice(value,options)).filter(Boolean).join(' · ')||'백분율은 소수 첫째 자리로 반올림합니다. 전체 원값은 아래에서 확인하세요.';
 $('condition-denominator').textContent=task==='verification'?`실제 검증 쌍 ${metric(values?.n_pairs)}`:`실제 검색 유효 질의 ${metric(values?.eligible_queries)} · 미측정 수량은 채워 넣지 않습니다.`;
 $('condition-rule').textContent='원본·blur0·mosaic0·mosaic1은 원본 결과 재사용 대상입니다. 재사용 사실과 실제 완료 여부는 아래 원기록을 따릅니다. PrivacyScore@1은 고정 공격의 1순위 재식별 실패율이며 익명성 보장이 아닙니다.';
 $('condition-source').textContent=s.run_id?`7차/runs/${s.run_id}/${row?.source||'privacy_results.json'} · ${dataset}/${selected.id}/${task} · ${row?.measurement_source||'미측정'}`:'현재 실행 ID를 확인할 수 없습니다.';
 $('condition-raw').textContent=JSON.stringify({dataset,task,condition:selected.id,status:state.status,source:row?.source??null,error:row?.error??null,stored_values:row?.raw??null},null,2);
 // 사진은 현재 실행의 고정 이름으로만 요청합니다. 사진 유무와 평가 완료는 별개입니다.
 for(const plan of EXPERIMENT_PLAN){
  const condition=CONDITIONS.find(c=>c.stage===plan.stage&&c.strength===selectedStrengths[plan.stage]);
  const image=$('preview-'+plan.mode),available=s.previews?.[dataset]?.[condition.id];
  const src=`/api/preview/${dataset}/${condition.id}.png`;
  if(available){if(image.dataset.run!==s.run_id||image.getAttribute('src')!==src){image.hidden=true;image.nextElementSibling.hidden=false;image.onload=()=>{image.hidden=false;image.nextElementSibling.hidden=true;};image.onerror=()=>{image.hidden=true;image.nextElementSibling.hidden=false;};image.dataset.run=s.run_id;image.src=src;}}
  else{image.hidden=true;image.removeAttribute('src');image.nextElementSibling.hidden=false;image.nextElementSibling.textContent=`${dataset} · ${condition.id} 대표사진 대기`;}
  const stage=document.querySelector(`.stage[data-stage="${plan.stage}"]`);const rows=(s.conditions||[]).filter(r=>r.dataset===dataset&&r.task===task&&r.stage===plan.stage);
  stage.classList.toggle('done',rows.length===plan.strengths.length&&rows.every(r=>r.status==='complete'));stage.classList.toggle('active',rows.some(r=>r.status==='observed'));stage.classList.toggle('failed',rows.some(r=>r.status==='error'));
  stage.querySelector('.stage-status').textContent=`${rows.filter(r=>r.status==='complete').length}/${plan.strengths.length}조건 결과 기록 · ${dataset} ${task==='verification'?'검증':'검색'}`;
 }
}
function renderLog(s){
 const names=logTab==='training'?['training']:['evaluation','privacy'];
 $('terminal').textContent=names.map(name=>`[${name}.stdout.log]\n${s.logs?.[name]?.text||'아직 기록이 없습니다.'}`).join('\n\n');
 $('log-info').textContent=names.map(name=>`${name}: ${s.logs?.[name]?.selection||'기록 대기'} · 원본 ${s.logs?.[name]?.bytes??0} B${s.logs?.[name]?.truncated?' · 최근 64 KiB만 표시':''}`).join(' / ');
}
function render(s){
 latest=s;$('status').textContent=describeExperiment(s).status;$('revision-label').textContent='화면 9차 · 실험 7차 · 읽기 전용';
 $('gpu').textContent=s.gpu.name;$('gpu-detail').textContent=s.gpu.utilization===null?'장치 조회 불가 · 학습 가능 여부를 뜻하지 않습니다.':`사용률 ${s.gpu.utilization}% · VRAM ${s.gpu.used} / ${s.gpu.total} MB`;
 $('result-count').textContent=`${s.completed_task_conditions}/84개 과제·조건`;$('download-summary').textContent='NIH + COVID19 · 저자 원본 목록';
 const display=trainingDisplay(s),training=display.progress,names={verification:'검증 모델',retrieval_phase1:'검색 출력부',retrieval_phase2:'검색 전체',evaluation:'원본 성능 평가',privacy:'변환 조건 평가'};
 $('training-state').textContent=training?`${display.lastObserved?'오류 전 마지막 관측 · ':''}${names[training.phase]||'진행 기록'} · 에폭 ${metric(training.epoch)} / ${metric(training.epochs)}`:display.terminal?'오류·중단 상태 · 정상 학습 기록 확인 불가':'학습 기록 대기';
 $('training-detail').textContent=training?`배치 ${metric(training.step)} / ${metric(training.steps)} · 단계 내 누적 처리 배치 ${metric(training.global_step)} · 손실 ${metric(training.loss)} · 학습률 ${metric(training.learning_rate)} · ${observedTime(training.updated_at)} · ${display.lastObserved?'events.jsonl 마지막 정상 기록':'progress.json 기록'} · ${s.process?.verified?'실제 프로세스 확인 · PID '+s.process.proofs[0].pid:'실행 프로세스 신원 미확인'}`:'현재 기록이 없다는 것이 0회 학습을 뜻하지 않습니다.';
 $('training-detail').textContent+=' · 처리 배치 수는 가중치 적용 횟수가 아닙니다. 혼합정밀도에서 값이 유효하지 않으면 해당 가중치 적용을 생략할 수 있습니다.';
 $('training-raw').textContent=JSON.stringify({current_progress_raw:s.progress_raw,last_observed_progress:s.last_observed_progress??null,history:s.history??null,process:s.process??null},null,2);
 for(const phase of phaseTimeline(s)){
  const card=document.querySelector(`.training-phase[data-phase="${phase.name}"]`);card.classList.toggle('done',phase.complete);card.classList.toggle('active',!!phase.active);card.querySelector('.training-phase-state').textContent=phase.label;
  card.querySelector('.training-phase-detail').textContent=phase.progress?`에폭 ${metric(phase.progress.epoch)} / ${metric(phase.progress.epochs)} · 단계 내 누적 처리 배치 ${metric(phase.progress.global_step)} · ${observedTime(phase.progress.updated_at)}`:phase.proof.selected_weights?'선택 가중치 파일 있음 · 최종 성능 완료와 별개':'현재 실행의 단계 기록 대기';
  card.querySelector('pre').textContent=JSON.stringify({progress:phase.progress,...phase.proof},null,2);
 }
 const historyMessage=historyNotice(s.history);
 $('history-note').textContent=historyMessage.info;$('history-note').hidden=!historyMessage.info;
 const p=s.progress;$('phase').textContent=p?`${names[p.phase]||p.phase} · 저장 상태 ${p.status} · ${observedTime(p.updated_at)}`:'현재 실행 포인터를 확인하고 있습니다.';
 notice([p?.error,s.pointer_error?'7차 현재 실행 포인터 또는 경계를 확인할 수 없습니다.':null,...(s.errors||[]).map(e=>e.source+' · '+e.error),historyMessage.error].filter(Boolean).join('\n'));
 $('check-model').textContent=s.config?'✓':'○';$('check-data').textContent=s.preflight?'✓':'○';$('check-validation').textContent=s.preflight?'기록':'○';
 $('downloads').textContent=s.config?`seed ${metric(s.config.seed)} · 검증 작업자 ${metric(s.config.verification_workers)} · 검색 작업자 ${metric(s.config.retrieval_workers)} · ${s.config.protocol?.dataset_cleaning_policy||'정책 확인 전'} · 사전 검사 파일 존재는 전수 픽셀 해시 검증을 의미하지 않습니다.`:'7차 학습 설정 기록 대기';
 renderTrainingVisual(s);renderExplorer(s);renderLog(s);
 const tbody=$('result-rows'),expanded=new Set([...tbody.querySelectorAll('details[open]')].map(x=>x.dataset.key));tbody.replaceChildren();
 for(const ds of DATASETS)for(const condition of CONDITIONS){
  const verification=conditionState(s,ds,'verification',condition.id),retrieval=conditionState(s,ds,'retrieval',condition.id);
  if(!verification.row?.raw&&!retrieval.row?.raw)continue;
  const v=verification.row?.metrics,r=retrieval.row?.metrics,tr=document.createElement('tr');
  const cells=[ds,condition.id,metric(v?.auc),formatPercent(v?.accuracy),formatPercent(r?.precision_at_1),formatPercent(r?.map_at_r),formatPercent(r?.privacy_score_at_1,{ratio:false}),`${verification.label} / ${retrieval.label}`];
  for(const value of cells){const td=document.createElement('td');td.textContent=value;tr.append(td);}
  for(const [column,value,options] of [[3,v?.accuracy,{}],[4,r?.precision_at_1,{}],[5,r?.map_at_r,{}],[6,r?.privacy_score_at_1,{ratio:false}]]){const text=percentNotice(value,options);if(text){const note=document.createElement('small');note.className='rounding-note';note.textContent=text;tr.children[column].append(note);}}
  const td=document.createElement('td'),details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');details.className='raw-values';details.dataset.key=ds+'/'+condition.id;details.open=expanded.has(details.dataset.key);summary.textContent='원값';pre.textContent=JSON.stringify({verification:verification.row,retrieval:retrieval.row},null,2);details.append(summary,pre);td.append(details);tr.append(td);tbody.append(tr);
 }
 $('empty').hidden=!!tbody.children.length;$('updated').textContent='최근 확인 '+observedTime(s.updated_at);
}
async function refresh(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('상태 조회 실패');render(await response.json());}catch(error){$('status').textContent='서버 연결 확인 필요';notice(error.message);}}
refresh();setInterval(refresh,3000);

function setProgress(id,ratio,label){
 const node=$(id);node.classList.toggle('unknown',ratio===null);
 node.querySelector('span').style.width=ratio===null?'0%':String(ratio*100)+'%';
 if(ratio===null){node.removeAttribute('aria-valuenow');node.setAttribute('aria-valuetext','미측정');}
 else{node.setAttribute('aria-valuenow',String(ratio*100));node.setAttribute('aria-valuetext',label);}
}
function renderTrainingVisual(s){
 const display=trainingDisplay(s),p=display.progress,f=trainingFractions(p);
 $('progress-context').textContent=display.lastObserved?'오류·중단 전 마지막 정상 관측':s.fresh&&s.process?.verified?'현재 관측 · 실제 프로세스 확인':s.fresh?'최근 관측 · 실행 신원 미확인':'과거 관측 · 현재 실행 미확인';
 $('progress-title').textContent=p?`에폭 ${metric(p.epoch)} / ${metric(p.epochs)}`:'학습 기록 대기';
 $('epoch-percent').textContent=formatProgressPercent(p,'epoch');$('phase-percent').textContent=formatProgressPercent(p,'phase');
 setProgress('epoch-progress',f?.epoch??null,formatProgressPercent(p,'epoch'));setProgress('phase-progress',f?.phase??null,formatProgressPercent(p,'phase'));
 $('epoch-count').textContent=f?`${p.step.toLocaleString('ko-KR')} / ${p.steps.toLocaleString('ko-KR')} 배치 처리 · 이 에폭에 남은 배치 ${f.remaining.toLocaleString('ko-KR')} · ${observedTime(p.updated_at)}`:'현재 배치·전체 배치 분모를 확인할 수 없습니다. 0%로 대체하지 않습니다.';
 drawLoss('loss-chart','loss-caption',s.history?.loss_samples,'loss',p?.phase);
 drawLoss('validation-chart','validation-caption',s.history?.validation_samples,'validation_loss',p?.phase);
 $('training-raw').textContent=JSON.stringify({current_progress_raw:s.progress_raw,last_observed_progress:s.last_observed_progress??null,progress_fractions:f,history:s.history??null,process:s.process??null},null,2);
}
function drawLoss(id,captionId,samples,key,phase){
 const svg=$(id);svg.replaceChildren();const selected=(samples||[]).filter(p=>p.phase===phase),segments=lossSegments(selected,key),points=segments.flat();
 const caption=$(captionId);caption.textContent=points.length?`${phase==='verification'?'검증 모델':phase==='retrieval_phase1'?'검색 출력부':'검색 전체'} · 정상 관측 ${points.length}개 · 마지막 ${observedTime(points.at(-1).updated_at)} · 최근 256 KiB 범위 · 재개/에폭 경계에서 선 분리`:key==='loss'?'최근 256 KiB 범위에 이 단계의 정상 학습 손실 기록이 없습니다.':'최근 256 KiB 범위에 이 단계의 에폭 완료·검증 손실 기록이 없습니다.';
 function element(name,attributes,text){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attributes))node.setAttribute(k,String(v));if(text!==undefined)node.textContent=text;svg.append(node);return node;}
 if(!points.length){element('text',{x:360,y:120,'text-anchor':'middle',class:'chart-label'},'기록 대기 · 미측정');return;}
 const xs=points.map(p=>p.x),ys=points.map(p=>p.value),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const x=v=>68+(v-xmin)/(xmax-xmin||1)*632,y=v=>190-(v-ymin)/(ymax-ymin||1)*160;
 element('path',{d:'M68 24V190H700',class:'chart-axis'});
 for(const [v,pos] of [[ymax,30],[ymin,190]])element('text',{x:60,y:pos,'text-anchor':'end',class:'chart-label'},Number(v).toPrecision(4));
 element('text',{x:68,y:220,class:'chart-label'},String(xmin));element('text',{x:700,y:220,'text-anchor':'end',class:'chart-label'},String(xmax));
 for(const group of segments){
  if(group.length>1)element('polyline',{points:group.map(p=>x(p.x)+','+y(p.value)).join(' '),class:'chart-line'});
  for(const point of group){const circle=element('circle',{cx:x(point.x),cy:y(point.value),r:3,class:'chart-point'});const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=`${key==='loss'?'처리 배치':'에폭'} ${point.x} · 손실 ${point.value} · ${observedTime(point.updated_at)}`;circle.append(title);}
 }
}
