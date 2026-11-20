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


export const DATASETS=Object.freeze(['NIH','COVID19']);
export const TASKS=Object.freeze(['verification','retrieval']);
export const LEVELS=Object.freeze([0,1,2,4,8,16,32,64,128,256]);
export const EXPERIMENT_PLAN=Object.freeze([
 {stage:1,mode:'original',title:'원본 평가',strengths:[0]},
 {stage:2,mode:'blur',title:'블러 평가',strengths:LEVELS},
 {stage:3,mode:'mosaic',title:'모자이크 평가',strengths:LEVELS}
]);
export const CONDITIONS=Object.freeze(EXPERIMENT_PLAN.flatMap(plan=>plan.strengths.map(strength=>Object.freeze({stage:plan.stage,mode:plan.mode,strength,id:plan.mode+'_'+String(strength).padStart(3,'0')}))));
export function measured(task,values){
 const keys=task==='verification'?['auc','accuracy']:['precision_at_1','map_at_r','r_precision'];
 return !!values&&keys.every(key=>{const value=values[key];return (typeof value==='number'||typeof value==='string'&&value.trim()!=='')&&Number.isFinite(Number(value))&&Number(value)>=0&&Number(value)<=1;});
}
export function conditionState(s,dataset,task,condition){
 const matches=(s.conditions||[]).filter(row=>row.dataset===dataset&&row.task===task&&row.condition===condition);
 if(matches.length!==1)return {status:matches.length?'error':'waiting',label:matches.length?'중복 기록 · 확인 필요':'평가 대기 · 미측정',row:null};
 const row=matches[0];
 return {...row,label:{complete:'평가 결과 기록',waiting:'평가 대기 · 미측정',error:'결과 기록 확인 필요',observed:'이 조건의 최근 진행 기록 · 프로세스 미확인'}[row.status]||'확인 필요',row};
}
export function trainingDisplay(s){
 const terminal=['failed','stopped','interrupted'].includes(s.progress?.status)||['failed','stopped'].includes(s.progress?.phase);
 return {progress:terminal?s.last_observed_progress:s.progress,lastObserved:terminal&&!!s.last_observed_progress,terminal};
}
export function trainingPhaseStates(s){
 const display=trainingDisplay(s),names=['verification','retrieval_phase1','retrieval_phase2'];
 return names.map(name=>({name,progress:display.progress?.phase===name?display.progress:null,lastObserved:display.lastObserved,
  active:!display.terminal&&s.fresh&&s.process?.verified&&s.process.proofs?.[0]?.script==='train_from_scratch.py'&&s.progress?.phase===name,complete:false,proof:s.training_phases?.[name]||{},
  label:display.progress?.phase===name?(display.terminal?'오류 전 마지막 학습 관측':s.fresh&&s.process?.verified&&s.process.proofs?.[0]?.script==='train_from_scratch.py'?'모델 학습 중 · 실제 프로세스 확인':s.fresh?'최근 학습 기록 · 프로세스 미확인':'이전 학습 기록 · 현재 실행 미확인'):s.training_phases?.[name]?.selected_weights?'선택 가중치 기록 있음':'학습 기록 대기'}));
}
export function describeExperiment(s){
 const p=s.progress;
 if(s.pointer_error)return {status:'현재 실행 연결 확인 필요'};
 if(!p&&s.completed_task_conditions!==84)return {status:s.process?.verified?'재개 프로세스 확인 · 진행 기록 대기':'현재 실행 기록 대기'};
 if(p?.status==='failed'||p?.phase==='failed')return {status:s.process?.verified?'재개 프로세스 확인 · 이전 오류 기록 유지':'실험 오류'};
 if(p?.status==='stopped'||p?.phase==='stopped')return {status:'실험 중단 기록'};
 if(s.completed_task_conditions===84)return {status:'84개 과제·조건 결과 기록'};
 if(s.fresh&&s.process?.verified){const script=s.process.proofs?.[0]?.script;return {status:script==='train_from_scratch.py'&&['verification','retrieval_phase1','retrieval_phase2'].includes(p?.phase)?'모델 학습 중 · 프로세스·최근 기록 확인':script==='cxr7_privacy.py'&&p?.phase==='privacy'||script==='cxr7_evaluate.py'&&p?.phase==='evaluation'?'평가 중 · 프로세스·최근 기록 확인':'재개 프로세스 확인 · 단계 기록 확인 중'};}
 if(s.fresh)return {status:'최근 진행 기록 · 실행 프로세스 미확인'};
 return {status:'이전 진행 기록 · 현재 실행 미확인'};
}

// 정상적인 읽기 범위 안내와 기록 손상 오류를 화면에서 구분합니다.
export function historyNotice(history){
 if(!history)return {info:'',error:''};
 const messages=[];
 if(history.truncated||history.status==='truncated')messages.push('학습 이력은 최근 256 KiB만 확인했습니다. 앞부분은 생략했으며 원본 파일은 보존됩니다.');
 if(history.incomplete_tail||history.status==='incomplete_tail')messages.push('작성 중인 마지막 행은 제외하고 완성된 기록만 확인했습니다.');
 if(history.status==='missing')messages.push('현재 실행의 학습 이력 파일을 기다립니다.');
 const errors={
  malformed:'학습 이력에 잘못된 JSON 또는 UTF-8 기록이 있어 마지막 관측값을 채택하지 않았습니다.',
  unreadable:'학습 이력 파일을 읽을 수 없어 마지막 관측값을 확인하지 못했습니다.',
  invalid_record:'학습 이력의 기록 형식이 잘못되어 마지막 관측값을 채택하지 않았습니다.',
  different_run:'학습 이력에서 다른 실행 ID를 발견해 마지막 관측값을 채택하지 않았습니다.',
  invalid_training_record:'학습 이력의 수치 또는 단계 검증에 실패해 마지막 관측값을 채택하지 않았습니다.'
 };
 const normal=['ok','missing','truncated','incomplete_tail'];
 return {info:messages.join(' '),error:normal.includes(history.status)?'':errors[history.status]||'학습 이력의 검사 상태를 확인할 수 없습니다.'};
}

// 분모와 범위를 확인한 관측만 진행 막대에 사용합니다. 미지 값은 0으로 바꾸지 않습니다.
export function trainingFractions(record){
 const valid=(x,min=0)=>Number.isSafeInteger(x)&&x>=min;
 if(!record||!valid(record.epoch,1)||!valid(record.epochs,1)||record.epoch>record.epochs||
    !valid(record.step)||!valid(record.steps,1)||record.step>record.steps)return null;
 return {epoch:record.step/record.steps,phase:((record.epoch-1)+record.step/record.steps)/record.epochs,
  remaining:record.steps-record.step,raw:{epoch:record.epoch,epochs:record.epochs,step:record.step,steps:record.steps}};
}
// 단계·에폭·시간·처리 순서가 되감기면 선을 끊어 재개 구간을 이어 그리지 않습니다.
export function lossSegments(samples,key='loss'){
 const segments=[];let previous=null;
 for(const sample of samples||[]){
  const time=Date.parse(sample.updated_at),value=sample[key];
  if(typeof value!=='number'||!Number.isFinite(value)||!Number.isFinite(time)) {previous=null;continue;}
  const x=key==='validation_loss'?sample.epoch:sample.global_step;
  if(!Number.isSafeInteger(x)||x<0){previous=null;continue;}
  if(!previous||previous.phase!==sample.phase||sample.epoch<previous.epoch||time<=Date.parse(previous.updated_at)||
      x<=(key==='validation_loss'?previous.epoch:previous.global_step)||key==='loss'&&sample.epoch!==previous.epoch)segments.push([]);
  segments.at(-1).push({...sample,x,value});previous=sample;
 }
 return segments;
}
export function phaseTimeline(s){
 const names=['verification','retrieval_phase1','retrieval_phase2'],display=trainingDisplay(s),current=names.indexOf(display.progress?.phase);
 return trainingPhaseStates(s).map((phase,index)=>{
  const laterObserved=s.config?.tasks==='both'&&current>index;
  const allDone=s.progress?.phase==='complete'&&s.progress?.status==='completed'&&s.config?.tasks==='both';
  const complete=(laterObserved||allDone)&&phase.proof.selected_weights===true;
  return {...phase,complete,label:complete?'학습 단계 완료 · 후속 단계/종료 기록 확인':phase.label};
 });
}

// 나눗셈으로 생기는 이진 부동소수 오차를 거치지 않고 원정수에서 0.1% 단위로 반올림합니다.
export function formatProgressPercent(record,kind='epoch'){
 if(!trainingFractions(record)||!['epoch','phase'].includes(kind))return '미측정';
 const step=BigInt(record.step),steps=BigInt(record.steps);
 const numerator=kind==='epoch'?step:(BigInt(record.epoch)-1n)*steps+step;
 const denominator=kind==='epoch'?steps:BigInt(record.epochs)*steps;
 const units=(numerator*2000n+denominator)/(denominator*2n);
 const digits=String(units).padStart(2,'0');return digits.slice(0,-1)+'.'+digits.slice(-1)+'%';
}

export const INDEPENDENT_CONDITIONS=Object.freeze(['blur','mosaic'].flatMap(mode=>LEVELS.map(strength=>Object.freeze({id:mode+'_'+String(strength).padStart(3,'0'),mode,strength,identity:mode==='blur'&&strength===0||mode==='mosaic'&&strength<=1}))));
export function independentLabel(status){return {waiting_original:'원본 전체 완료 대기',pending:'순차 실행 대기',training:'학습 중 · 실제 관측',validating:'에폭 검증 중 · 실제 프로세스 확인',waiting_gpu:'GPU 여유 대기 · 학습 배치 전',preparing:'실행 준비 · 학습 미확인',complete:'독립 학습·평가 완료',failed:'오류 · 다음 조건 중단',stopped:'중단 기록',evaluating:'동일 조건 평가 기록',blocked:'조건 확인 필요 · 실행 차단',stale:'학습 기록 · 현재 실행 미확인',unknown:'기록 대기 · 미측정'}[status]||'기록 확인 필요';}
