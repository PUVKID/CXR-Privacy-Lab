// 현재 실행의 기록만 사용하며 준비 과정과 평가 과정을 구분합니다.
export function describeExperiment(s) {
 const run=s.run || {};const log=(s.log?.text || '').split('[새 실행]').at(-1);
 const matches=[...log.matchAll(/\[단계 ([123])\]/g)];
 const lastStage=Number(matches.at(-1)?.[1] || 0);
 const evaluating=/평가 실행/.test(run.phase || '');
 const labels={completed:'평가 완료',failed:'실험 오류',stopped:'실험 중단',interrupted:'실행 중단됨'};
 const status=s.starting?'실험 시작 중':run.status==='running'?(evaluating?'평가 실행 중':'실험 준비 중'):(labels[run.status] || '실험 대기');
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
