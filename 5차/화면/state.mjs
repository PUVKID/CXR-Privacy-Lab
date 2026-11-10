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
 if(parsed.digits==='0')return '0.00%';
 // 원비율은 백분율로 두 자리 이동하고, 표시 소수 두 자리만 정수 문자열로 남깁니다.
 const shift=parsed.exponent-BigInt(parsed.fraction)+BigInt(ratio?4:2);
 let units;
 if(shift>=0n)units=parsed.digits+'0'.repeat(Number(shift));
 else{const keep=BigInt(parsed.digits.length)+shift;units=keep<=0n?'0':parsed.digits.slice(0,Number(keep));}
 units=units.padStart(3,'0');
 const sign=parsed.negative&&/[1-9]/.test(units)?'-':'';
 return `${sign}${units.slice(0,-2)}.${units.slice(-2)}%`;
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
  const count=s.results.filter(r=>r.source.startsWith(`stage_${n}_`)).length;
  const done=count>=(n===1?1:10);
  const failed=run.status==='failed'&&lastStage===n;
  const active=s.active&&run.status==='running'&&evaluating&&lastStage===n;
  const stopped=['stopped','interrupted'].includes(run.status)&&lastStage===n;
  return {done,failed,active,label:failed?'평가 실패':done?'평가 완료':active?'평가 중':stopped?'중단됨':run.status==='failed'&&!lastStage?'준비 실패로 대기':count?`${count}개 조건 기록`:'평가 대기'};
 });
 return {status,stages};
}
