// CSV 십진 문자열을 보존하며 Number는 비유한 값 거부에만 사용합니다.
function decimal(value) {
 if(typeof value!=='string'&&typeof value!=='number')return null;
 const raw=String(value),text=raw.trim();
 const match=/^([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:[eE]([+-]?\d+))?$/.exec(text);
 if(!match||!Number.isFinite(Number(text)))return null;
 const fraction=match[3]??match[4]??'';
 return {raw,negative:match[1]==='-',digits:((match[2]??'0')+fraction).replace(/^0+/,'')||'0',fraction:fraction.length,exponent:BigInt(match[5]??'0')};
}

export function formatRaw(value) {
 return decimal(value)?.raw??'미측정';
}

export function formatPercent(value,{ratio=true}={}) {
 const parsed=decimal(value);if(!parsed)return '미측정';
 if(parsed.digits==='0')return '0.0%';
 // 십진 문자열에서 백분율 소수 첫째 자리까지 이동하고 다음 자리로 반올림합니다.
 const shift=parsed.exponent-BigInt(parsed.fraction)+BigInt(ratio?3:1);
 let units;
 if(shift>=0n)units=parsed.digits+'0'.repeat(Number(shift));
 else{
  const keep=BigInt(parsed.digits.length)+shift;
  units=keep<=0n?'0':parsed.digits.slice(0,Number(keep));
  const next=keep<0n?'0':parsed.digits[Number(keep)]??'0';
  if(next>='5')units=String(BigInt(units)+1n);
 }
 units=units.padStart(2,'0');
 const sign=parsed.negative&&/[1-9]/.test(units)?'-':'';
 return `${sign}${units.slice(0,-1)}.${units.slice(-1)}%`;
}

export function percentNotice(value,{ratio=true}={}) {
 if(formatPercent(value,{ratio})!=='100.0%')return '';
 const p=decimal(value);let digits=p.digits,power=p.exponent-BigInt(p.fraction);
 while(digits.endsWith('0')){digits=digits.slice(0,-1);power++;}
 const targetPower=ratio?0n:2n;
 if(digits==='1'&&power===targetPower)return ratio?'저장 원비율은 정확히 1':'저장 점수는 정확히 100';
 const order=BigInt(digits.length)+power,targetOrder=1n+targetPower;
 const less=order<targetOrder||(order===targetOrder&&digits.padEnd(Math.max(digits.length,1),'0')<'1'.padEnd(digits.length,'0'));
 return less?'반올림 표시 · 저장값은 100% 미만':'반올림 표시 · 저장값은 100% 초과';
}

export const EXPERIMENT_PLAN=Object.freeze([
 {stage:1,mode:'original',source:'stage_1_original',title:'원본 평가',unit:'원본',strengths:[0],description:'변형하지 않은 시험 영상으로 기준 성능을 측정합니다.'},
 {stage:2,mode:'blur',source:'stage_2_blur',title:'블러 평가',unit:'Gaussian radius · 원영상 px',strengths:[0,2,4,8,16,32,64,128,256,512],description:'질의 영상에만 가우시안 블러를 적용합니다. 반경은 원영상 픽셀 기준입니다.'},
 {stage:3,mode:'mosaic',source:'stage_3_mosaic',title:'모자이크 평가',unit:'목표 타일 크기 · px',strengths:[1,4,8,16,32,64,128,256,512,1024],description:'질의 영상만 목표 타일 크기로 모자이크 처리합니다.'},
].map(p=>Object.freeze({...p,strengths:Object.freeze(p.strengths)})));

export function lastCondition(s) {
 const log=(s.log?.text||'').split('[새 실행]').at(-1);
 const matches=[...log.matchAll(/\[단계 ([123])\]\s+(original|blur|mosaic)\s*\/\s*강도 (\d+)(?=\s*(?::|$))/gm)];
 const match=matches.at(-1);return match?{stage:Number(match[1]),mode:match[2],strength:Number(match[3])}:null;
}

export function conditionState(s,stage,strength) {
 const plan=EXPERIMENT_PLAN.find(p=>p.stage===stage);
 if(!plan||!plan.strengths.includes(strength))return {status:'error',label:'정의되지 않은 조건',row:null,rows:[]};
 const rows=(s.results||[]).filter(r=>r.source===plan.source&&/^\d+(?:\.0+)?$/.test(String(r.strength))&&Number(r.strength)===strength);
 if(rows.length>1)return {status:'error',label:'중복 결과 · 원값 확인 필요',row:null,rows};
 if(rows.length===1){
  const row=rows[0];
  if((row.stage!==undefined&&Number(row.stage)!==stage)||(row.mode!==undefined&&row.mode!==plan.mode))return {status:'error',label:'결과 조건 불일치',row:null,rows};
  return {status:'complete',label:'평가 결과 기록',row,rows};
 }
 const last=lastCondition(s),run=s.run||{};
 const exact=last?.stage===stage&&last?.strength===strength&&last?.mode===plan.mode;
 const evaluating=/평가 실행/.test(run.phase||'');
 if(exact&&evaluating&&run.status==='failed')return {status:'error',label:'이 조건 평가 실패',row:null,rows};
 if(exact&&evaluating&&['stopped','interrupted'].includes(run.status))return {status:'stopped',label:'이 조건에서 중단',row:null,rows};
 if(exact&&evaluating&&s.active&&run.status==='running')return {status:'running',label:'평가 실행 중 · 현재 로그 기준',row:null,rows};
 return {status:'waiting',label:'평가 대기 · 미측정',row:null,rows};
}

export function trainingPhaseStates(s) {
 const names=['verification','retrieval_phase1','retrieval_phase2'];
 const terminal=['failed','stopped','interrupted'].includes(s.run?.status);
 const observed=terminal?s.last_observed_progress:null;
 const current=names.indexOf(observed?.phase??s.training?.phase);
 return names.map((name,index)=>{
  const proof=s.training_phases?.[name]||{};
  const complete=proof.selected_weights&&(s.training_artifacts_complete||current>index);
  const active=current===index&&!terminal&&s.active&&s.run?.status==='running';
  const phaseStatus={waiting_for_gpu:'GPU 여유 메모리 대기',initializing:'새 모델 초기화 중',validating:'학습 모델 검증 중',epoch_complete:'에폭 완료',amp_backoff:'유한 기울기 재확인 중',training:s.training?.global_step===0?'첫 가중치 갱신 대기':'모델 학습 중'}[s.training?.status]||'현재 단계 진행 중';
  const label=complete?'학습 완료 · 다음 단계 기록 확인':current===index?(['failed','stopped','interrupted'].includes(s.run?.status)?'학습 중단 또는 오류':active?phaseStatus:'현재 단계 기록'):proof.first_backward?'가중치 갱신 기록 있음':proof.initialization?'초기화 기록 있음':'학습 대기';
  return {name,active,complete:!!complete,label,proof,progress:current===index?(terminal?observed:s.training):null,lastObserved:current===index&&!!observed};
 });
}

// 현재 실행의 기록만 사용하며 준비 과정과 평가 과정을 구분합니다.
export function describeExperiment(s) {
 const run=s.run || {};const log=(s.log?.text || '').split('[새 실행]').at(-1);
 const matches=[...log.matchAll(/\[단계 ([123])\]/g)];
 const lastStage=Number(matches.at(-1)?.[1] || 0);
 const evaluating=/평가 실행/.test(run.phase || '');
 const training=/모델.*학습/.test(run.phase || '');
 const labels={completed:run.mode==='데이터 준비 전용'?'데이터 준비 완료':'학습·평가 완료',failed:'실험 오류',stopped:'실험 중단',interrupted:'실행 중단됨'};
 const trainingStatus=!s.training?'학습 시작 준비 중':s.training.phase==='data_validation'?'학습 데이터 검증 중':s.training.status==='waiting_for_gpu'?'GPU 여유 메모리 대기':s.training.status==='initializing'?'새 모델 초기화 중':s.training.status==='validating'?'학습 모델 검증 중':s.training.global_step===0?'첫 가중치 갱신 대기':'새 모델 학습 중';
 const status=s.starting?'실험 시작 중':run.status==='running'?(evaluating?'평가 실행 중':training?trainingStatus:'실험 준비 중'):(labels[run.status] || '실험 대기');
 const stages=[1,2,3].map(n=>{
  const conditions=EXPERIMENT_PLAN[n-1].strengths.map(strength=>conditionState(s,n,strength));
  const count=conditions.filter(c=>c.status==='complete').length;
  const done=conditions.every(c=>c.status==='complete');
  const failed=conditions.some(c=>c.status==='error')||(run.status==='failed'&&lastStage===n);
  const active=s.active&&run.status==='running'&&evaluating&&lastStage===n;
  const stopped=['stopped','interrupted'].includes(run.status)&&lastStage===n;
  return {done,failed,active,label:failed?'평가 실패 또는 결과 확인 필요':done?`${count}/${conditions.length}개 조건 결과 기록`:active?'평가 중':stopped?'중단됨':run.status==='failed'&&!lastStage?'준비 실패로 대기':count?`${count}/${conditions.length}개 조건 기록`:'평가 대기'};
 });
 return {status,stages};
}

// 오류 원값과 마지막 정상 학습 기록을 분리하며 현재 실행 중인 값이 우선입니다.
export function trainingDisplay(s) {
 const terminal=['failed','stopped','interrupted'].includes(s.run?.status);
 return { progress:terminal?(s.last_observed_progress??null):(s.training??null), lastObserved:terminal&&!!s.last_observed_progress, terminal };
}
