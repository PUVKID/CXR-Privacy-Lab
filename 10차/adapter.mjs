import { promises as fs } from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import { parseCSV, readTrainingHistory, tail } from './io.mjs';
import { CONDITIONS, DATASETS, TASKS, measured, INDEPENDENT_CONDITIONS } from './화면/state.mjs';

export const validRunId=id=>typeof id==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
// 포인터의 path와 결과 안의 image_path는 읽기 경로로 사용하지 않습니다.
export async function boundedFile(base,relative,limit=2*1024*1024){
 const target=path.join(base,relative);
 try{
  const realBase=await fs.realpath(base),real=await fs.realpath(target);
  if(real!==path.resolve(realBase,relative))throw Error('파일 연결 경계 불일치');
  const info=await fs.stat(real);if(!info.isFile()||info.size>limit)throw Error('파일 크기 또는 형식 제한 초과');
  const data=await fs.readFile(real);if(data.length>limit)throw Error('파일 크기 제한 초과');
  return {text:new TextDecoder('utf-8',{fatal:true}).decode(data),bytes:data.length};
 }catch(error){if(error.code==='ENOENT')return null;throw error;}
}
async function json(base,relative,errors,limit){
 try{const data=await boundedFile(base,relative,limit);return data?{value:JSON.parse(data.text.replace(/^\uFEFF/,'')),text:data.text}:null;}
 catch(error){errors.push({source:relative,error:'기록 읽기 또는 JSON 검증 실패'});return null;}
}
export function previewName(condition){
 const item=CONDITIONS.find(c=>c.id===condition);if(!item)return null;
 const index=CONDITIONS.indexOf(item);
 return String(index).padStart(2,'0')+'_'+(item.mode==='original'?'original':item.mode+'_'+String(item.strength).padStart(3,'0'))+'.png';
}
export function adaptConditions(report,csv,progress,fresh,baseline=null,artifacts=null){
 const conditions=[];
 for(const dataset of DATASETS)for(const task of TASKS)for(const plan of CONDITIONS){
  const matches=(csv||[]).filter(row=>row.dataset===dataset&&row.condition===plan.id);
  const saved=report?.results?.[dataset]?.[plan.id];
  let metrics=null,status='waiting',source='privacy_results.json',raw=saved??null,error=null;
  if(matches.length>1){status='error';error='동일 데이터셋·조건 CSV 중복';raw=matches;}
  else if(matches.length===1){
   const row=matches[0];source='privacy_results.csv';raw=row;
   metrics=task==='verification'?{auc:row.verification_auc,accuracy:row.verification_accuracy,n_pairs:row.verification_n_pairs}:
    {precision_at_1:row.retrieval_precision_at_1,map_at_r:row.retrieval_map_at_r,r_precision:row.retrieval_r_precision,eligible_queries:row.retrieval_eligible_queries,privacy_score_at_1:row.privacy_score_at_1};
   if(row.mode!==plan.mode||String(row.strength)!==String(plan.strength)){status='error';error='CSV 조건 불일치';}
   else if(row[task+'_status']==='completed'){
    status=measured(task,metrics)?'complete':'error';
    if(saved&&saved[task+'_status']!=='completed'){status='error';error='CSV와 JSON 완료 상태 불일치';}
    if(saved&&measured(task,saved[task])&&Object.keys(task==='verification'?{auc:1,accuracy:1}:{precision_at_1:1,map_at_r:1,r_precision:1}).some(key=>Number(saved[task][key])!==Number(metrics[key]))){status='error';error='CSV와 JSON 지표 불일치';}
   }
  }else if(saved){
   metrics=saved[task]??null;
   if(saved.mode!==plan.mode||saved.strength!==plan.strength){status='error';error='JSON 조건 불일치';}
   else if(saved[task+'_status']==='completed')status=measured(task,metrics)?'complete':'error';
  }
  if(!matches.length&&!saved&&plan.mode==='original'&&baseline?.[dataset]?.[task]){
   source='evaluation_results.json';raw=baseline[dataset][task];
   const checkpoint=baseline[dataset][task+'_sha256'];
   if(artifacts?.published_cxr_initialization===false&&typeof checkpoint==='string'&&/^[a-f0-9]{64}$/i.test(checkpoint)&&checkpoint===artifacts[task+'_sha256']&&measured(task,raw)){
    status='complete';metrics=raw;
   }else{status='error';metrics=null;error='원본 평가의 지표 또는 현재 모델 SHA256 확인 실패';}
  }
  const exact=progress?.phase==='privacy'&&progress?.dataset===dataset&&progress?.task===task&&progress?.condition===plan.id;
  if(status==='waiting'&&exact&&fresh&&!['failed','stopped','completed'].includes(progress.status))status='observed';
  conditions.push({dataset,task,condition:plan.id,...plan,status,metrics:status==='complete'?metrics:null,raw,source,error,
    measurement_source:saved?.[task+'_measurement_source']??(plan.mode==='original'||plan.strength===0||plan.mode==='mosaic'&&plan.strength===1?'원본 재사용 대상 · 실제 측정 여부는 상태 확인':'질의 변환 대상 · 실제 측정 여부는 상태 확인')});
 }
 return conditions;
}
export async function readSevenStatus(executionRoot){
 const errors=[];
 const pointer=await json(path.join(executionRoot,'work'),'current_run.json',errors);
 const runId=pointer?.value?.run_id;
 const base={revision:10,read_only:true,experiment:{screen_revision:10,execution_revision:7,condition_count:21,task_condition_count:84,datasets:DATASETS,tasks:TASKS},run_id:validRunId(runId)?runId:null,pointer_error:!validRunId(runId),pointer_created_at:pointer?.value?.created_at??null,errors,process_verified:false,updated_at:new Date().toISOString()};
 if(!validRunId(runId))return {...base,progress:null,conditions:adaptConditions(null,null,null,false),completed_task_conditions:0,training_phases:{},logs:{},previews:{}};
 const folder=path.join(executionRoot,'runs',runId);
 try{if(await fs.realpath(folder)!==path.resolve(await fs.realpath(executionRoot),'runs',runId))throw Error('실행 경계 불일치');}
 catch(error){return {...base,pointer_error:true,errors:[...errors,{source:'runs/<현재 실행>',error:'실행 폴더 경계 확인 실패'}],progress:null,conditions:adaptConditions(null,null,null,false),completed_task_conditions:0,training_phases:{},logs:{},previews:{}};}
 const progressFile=await json(folder,'progress.json',errors),progress=progressFile?.value??null;
 const age=Date.now()-Date.parse(progress?.updated_at);
 const fresh=Number.isFinite(age)&&age>=-5000&&age<90000;
 let history={source:'events.jsonl',status:'missing',last_observed_progress:null};
 try{
  const file=path.join(folder,'events.jsonl');
  if(await fs.realpath(file)!==path.resolve(folder,'events.jsonl'))throw Error('기록 연결 경계 불일치');
  history=await readTrainingHistory(file,runId);
  history.source='events.jsonl';
 }catch(error){if(error.code!=='ENOENT'){history.status='unreadable';errors.push({source:'events.jsonl',error:'기록 경계 확인 실패'});}}
 const report=await json(folder,'privacy_results.json',errors),baseline=await json(folder,'evaluation_results.json',errors);
 let csv=null;
 try{const data=await boundedFile(folder,'privacy_results.csv');if(data)csv=parseCSV(data.text);}
 catch(error){errors.push({source:'privacy_results.csv',error:'CSV 읽기 실패'});csv=[];}
 const artifacts=await json(folder,'artifacts.json',errors);
 const conditions=adaptConditions(report?.value,csv,progress,fresh,baseline?.value,artifacts?.value);
 const training_phases={};
 for(const name of ['verification','retrieval_phase1','retrieval_phase2']){
  let selected_weights=false;
  try{selected_weights=(await fs.stat(path.join(folder,name,'selected_weights.pth'))).isFile();}catch(error){if(error.code!=='ENOENT')errors.push({source:name,error:'가중치 파일 정보 확인 실패'});}
  training_phases[name]={initialization:(await json(folder,name+'/initialization.json',errors))?.value??null,first_backward:(await json(folder,name+'/first_backward.json',errors))?.value??null,selected_weights};
 }
 const logs={};
 const names=await fs.readdir(folder);
 for(const name of ['training','evaluation','privacy']){
  const pattern=new RegExp('^'+name+'\\.stdout\\.log\\.(\\d{8}-\\d{6}-[0-9a-f]{6})\\.(stdout|stderr)\\.log$');
  const stamps=[...new Set(names.map(file=>pattern.exec(file)?.[1]).filter(Boolean))].sort();
  const latest=stamps.at(-1);
  const selected=latest?[name+'.stdout.log.'+latest+'.stdout.log',name+'.stdout.log.'+latest+'.stderr.log']:[name+'.stdout.log'];
  const parts=[];
  for(const selectedName of selected){
   try{const file=path.join(folder,selectedName),real=await fs.realpath(file);if(real!==path.resolve(folder,selectedName))throw Error('로그 연결 경계 불일치');parts.push({source:selectedName,...await tail(real)});}
   catch(error){if(error.code!=='ENOENT')errors.push({source:selectedName,error:'로그 읽기 실패'});}
  }
  logs[name]={text:parts.map(part=>'['+part.source+']\n'+part.text).join('\n'),bytes:parts.reduce((sum,part)=>sum+part.bytes,0),truncated:parts.some(part=>part.truncated),sources:parts.map(part=>part.source),selection:latest?'현재 실행의 가장 최근 재개 로그 쌍':'현재 실행의 고정 로그'};
 }
 const previews={};
 for(const dataset of DATASETS){previews[dataset]={};for(const condition of CONDITIONS){
  const relative=path.join('대표사진',dataset,previewName(condition.id));
  try{const real=await fs.realpath(path.join(folder,relative));const info=await fs.stat(real);previews[dataset][condition.id]=real===path.resolve(folder,relative)&&info.isFile()&&info.size<=12*1024*1024;}
  catch(error){previews[dataset][condition.id]=false;}
 }}
 return {...base,progress,progress_raw:progressFile?.text??null,fresh,history,last_observed_progress:history.last_observed_progress,conditions,
 completed_task_conditions:conditions.filter(row=>row.status==='complete').length,training_phases,logs,previews,
 config:(await json(folder,'training_config.json',errors))?.value??null,preflight:(await json(folder,'preflight.json',errors))?.value??null,
 baseline:baseline?.value??null,baseline_raw:baseline?.text??null};
}

export const INDEPENDENT_PROTOCOL='independent_strength_training_both_sides_v1';
const canonicalIndependent=value=>JSON.stringify((function ordered(v){return Array.isArray(v)?v.map(ordered):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,ordered(v[k])])):v;})(value));
export function independentDefinition(id){
 const p=INDEPENDENT_CONDITIONS.find(c=>c.id===id);if(!p)return null;
 return {id:p.id,mode:p.mode,strength:p.strength,identity_transform:p.identity,operator_mode:p.identity?'original':p.mode,operator_strength:p.identity?0:p.strength,
  units:p.mode==='blur'?'source_pixel_gaussian_standard_deviation':'source_pixel_tile_size',training_sides:'all_training_and_validation_images',evaluation_sides:'transformed_query_and_transformed_gallery',original_image_modified:false};
}
export function independentFingerprint(id){const condition=independentDefinition(id);return condition?createHash('sha256').update(canonicalIndependent({protocol:INDEPENDENT_PROTOCOL,condition,seed:3,initialization:'fresh_imagenet_seed3',cross_condition_state_inherited:false}),'utf8').digest('hex'):null;}
const QUEUE_STATUSES=['waiting_baseline','ready','training','evaluating','completed','failed','blocked'];
const ROW_STATUSES=['pending','training','evaluating','completed','failed','blocked'];
const queueLabel=status=>({waiting_baseline:'원본 전체 정상 완료 대기 · GPU 조건 학습 전',ready:'독립 학습 순차 실행 준비',training:'독립 조건 학습 기록',evaluating:'독립 조건 평가 기록',completed:'20조건 정상 완료 기록',failed:'큐 오류 · 다음 조건 중단',blocked:'실행 조건 확인 필요 · GPU 진입 차단'}[status]||'큐 기록 대기 · 미측정');
// 클라이언트 경로나 UUID를 받지 않고 정의된 조건에서 검증한 큐 행만 읽습니다.
async function independentQueue(root){
 const errors=[],file=await json(root,'work/queue/state.json',errors,1024*1024);
 if(!file)return {value:null,errors};
 const q=file.value;
 try{
  if(q?.schema_version!==1||q.protocol!==INDEPENDENT_PROTOCOL||(!validRunId(q.queue_id)||q.queue_id!==q.queue_id.toLowerCase())||!QUEUE_STATUSES.includes(q.status)||!Number.isFinite(Date.parse(q.updated_at))||
     !Array.isArray(q.conditions)||q.conditions.length!==20)throw Error();
  if(q.current_condition_id!==null&&!INDEPENDENT_CONDITIONS.some(c=>c.id===q.current_condition_id))throw Error();
  const ids=new Set();
  for(const row of q.conditions){
   const plan=INDEPENDENT_CONDITIONS.find(c=>c.id===row.id);
   if(!plan||ids.has(row.id)||row.mode!==plan.mode||row.strength!==plan.strength||row.identity_transform!==plan.identity||
      !ROW_STATUSES.includes(row.status)||row.initialization!=='fresh_imagenet_seed3'||row.condition_fingerprint!==independentFingerprint(row.id))throw Error();
   ids.add(row.id);
   if(row.run_id===null){if(row.training_subdir!==null||!['pending','blocked'].includes(row.status))throw Error();}
   else if(!validRunId(row.run_id)||row.run_id!==row.run_id.toLowerCase()||row.training_subdir!=='work/runs/'+row.run_id+'/training')throw Error();
  }
  const runIds=q.conditions.map(r=>r.run_id).filter(Boolean);if(new Set(runIds).size!==runIds.length)throw Error();
  const active=q.conditions.filter(r=>['training','evaluating'].includes(r.status));
  if(active.length>1||active.some(r=>r.id!==q.current_condition_id)||q.status==='waiting_baseline'&&q.conditions.some(r=>!['pending','blocked'].includes(r.status))||q.status==='completed'&&q.conditions.some(r=>r.status!=='completed'))throw Error();

  return {value:q,errors};
 }catch(error){return {value:null,errors:[...errors,{source:'work/queue/state.json',error:'독립 학습 큐의 조건·실행 경계 검증 실패'}]};}
}
export async function readIndependentOverview(root){
 const {value:q,errors}=await independentQueue(root);
 return {status:q?.status||'unknown',label:errors.length?'큐 기록 확인 필요 · 조건 조회 차단':queueLabel(q?.status),errors,
  conditions:INDEPENDENT_CONDITIONS.map(plan=>{const row=q?.conditions.find(r=>r.id===plan.id);return {...plan,status:row?row.status==='pending'?(q.status==='waiting_baseline'?'waiting_original':'pending'):row.status==='completed'?'complete':row.status==='training'?'stale':row.status==='evaluating'?'evaluating':row.status==='blocked'?'blocked':row.status:'unknown'};})};
}
export async function readIndependentCondition(root,id){
 const plan=INDEPENDENT_CONDITIONS.find(c=>c.id===id);if(!plan)return null;
 const {value:q,errors}=await independentQueue(root);
 const empty={condition:plan,status:'unknown',progress:null,progress_raw:null,last_observed_progress:null,history:null,initialization:null,training_phases:{},errors};
 if(!q)return empty;
 const row=q.conditions.find(r=>r.id===id);empty.queue_error=row.error??null;
 let status=row.status==='pending'?(q.status==='waiting_baseline'?'waiting_original':'pending'):row.status==='completed'?'complete':row.status==='training'?'stale':row.status;
 if(!row.run_id)return {...empty,status};
 const folder=path.resolve(root,'work/runs',row.run_id,'training');
 try{if(await fs.realpath(folder)!==folder)throw Error();}
 catch(error){return {...empty,status:'unknown',errors:[...errors,{source:'선택 조건 학습 기록',error:'실행 폴더 경계 또는 존재 확인 실패'}]};}
 const current=await json(folder,'progress.json',errors),config=await json(folder,'training_config.json',errors),artifacts=await json(folder,'artifacts.json',errors),evaluation=await json(folder,'evaluation_results.json',errors);
 const bound=value=>value&&value.condition_id===id&&value.condition_fingerprint===row.condition_fingerprint&&value.run_id===row.run_id&&
  value.initialization_mode==='fresh_imagenet_seed3'&&value.cross_condition_state_inherited===false&&value.experiment_protocol===INDEPENDENT_PROTOCOL&&
  canonicalIndependent(value.condition)===canonicalIndependent(independentDefinition(id));
 const canonical=value=>JSON.stringify((function ordered(v){return Array.isArray(v)?v.map(ordered):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,ordered(v[k])])):v;})(value));
 const sameDefinition=value=>bound(value)&&canonical(value.condition)===canonical(config?.value?.condition);
 if(!bound(config?.value))return {...empty,status:['failed','blocked'].includes(status)?status:'unknown',errors:[...errors,{source:'training_config.json',error:'선택 조건·실행·설정 지문 결합 확인 실패'}]};

 let history=null;
 try{const file=path.join(folder,'events.jsonl');if(await fs.realpath(file)!==file)throw Error();history=await readTrainingHistory(file,row.run_id);}
 catch(error){if(error.code!=='ENOENT')errors.push({source:'events.jsonl',error:'선택 조건 이력 경계 확인 실패'});}
 const phases={};
 for(const name of ['verification','retrieval_phase1','retrieval_phase2']){
  const initialization=(await json(folder,name+'/initialization.json',errors))?.value??null;
  let selected_weights=false;
  try{const file=path.join(folder,name,'selected_weights.pth');if(await fs.realpath(file)!==file)throw Error();const info=await fs.stat(file);selected_weights=info.isFile()&&info.size>0;}catch(error){if(error.code!=='ENOENT')errors.push({source:name,error:'체크포인트 경계 확인 실패'});}
  if(initialization&&!sameDefinition(initialization))return {...empty,status:'unknown',errors:[...errors,{source:name+'/initialization.json',error:'다른 조건 또는 실행의 초기화 기록 거부'}]};
  phases[name]={initialization,selected_weights};
 }
 if(artifacts&&!sameDefinition(artifacts.value)||evaluation&&!sameDefinition(evaluation.value))return {...empty,status:'unknown',errors:[...errors,{source:'선택 조건 평가',error:'결과의 조건·실행 지문 결합 확인 실패'}]};
 if(status==='complete'){
  const a=artifacts?.value,e=evaluation?.value;
  const valid=phases.verification.selected_weights&&phases.retrieval_phase2.selected_weights&&a?.published_cxr_initialization===false&&e?.completed===true&&['verification','retrieval'].every(task=>
   /^[a-f0-9]{64}$/i.test(a[task+'_sha256']||'')&&['NIH','COVID19'].every(dataset=>measured(task,e[dataset]?.[task])&&e[dataset]?.[task+'_sha256']===a[task+'_sha256']));
  if(!valid)return {...empty,status:'unknown',errors:[...errors,{source:'선택 조건 평가',error:'학습·동일 조건 NIH/COVID 평가 완료 결합 확인 실패'}]};
 }
 const age=Date.now()-Date.parse(current?.value?.updated_at),fresh=Number.isFinite(age)&&age>=-5000&&age<90000;
 return {...empty,status,progress:current?.value??null,progress_raw:current?.text??null,last_observed_progress:history?.last_observed_progress??null,history,config:config?.value??null,artifacts:artifacts?.value??null,evaluation:evaluation?.value??null,training_phases:phases,fresh,
  initialization_label:'계획: 새 ImageNet 초기값·시드3. 아래 단계별 초기화 원기록에서 실제 실행 여부를 확인합니다.',
  condition_fingerprint:row.condition_fingerprint,process_claim:row.process??null,run_id:row.run_id,errors};
}
