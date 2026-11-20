import test from 'node:test';
import assert from 'node:assert/strict';
import {promises as fs} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {fileURLToPath} from 'node:url';
import {createDashboard} from '../server.mjs';
import {readSevenStatus,adaptConditions,previewName,boundedFile} from '../adapter.mjs';
import {CONDITIONS,LEVELS,formatPercent,formatRaw,percentNotice,trainingDisplay,trainingPhaseStates,describeExperiment} from '../화면/state.mjs';
import {readTrainingHistory} from '../io.mjs';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const ID='12345678-1234-4234-8234-123456789abc';
const training={phase:'verification',status:'training',epoch:2,epochs:9,step:17,steps:40,global_step:57,loss:0.125,learning_rate:0.0025,updated_at:'2030-01-02T10:00:00.123456+00:00'};
async function fixture(fn){
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'cxr8-')),run=path.join(dir,'runs',ID),work=path.join(dir,'work');
 await fs.mkdir(run,{recursive:true});await fs.mkdir(work);
 const write=(name,value)=>fs.writeFile(path.join(run,name),typeof value==='string'?value:JSON.stringify(value));
 await fs.writeFile(path.join(work,'current_run.json'),JSON.stringify({run_id:ID,path:'../../untrusted'}));
 const server=createDashboard({root:ROOT,executionRoot:dir,processQuery:async()=>[],gpuQuery:(_a,_b,_c,cb)=>cb(null,'Test GPU, 0, 0, 24564')});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${server.address().port}`;
 try{await fn({dir,run,work,write,base,get:url=>fetch(base+url)});}finally{await new Promise(resolve=>server.close(resolve));await fs.rm(dir,{recursive:true,force:true});}
}
const metrics={verification:{auc:0.9940123456789012,accuracy:0.9878999999999999,n_pairs:100},retrieval:{precision_at_1:0.98,map_at_r:0.9,r_precision:0.91,eligible_queries:50,privacy_score_at_1:2}};
function reportRow(condition=CONDITIONS[0]){return {...condition,verification_status:'completed',retrieval_status:'completed',...metrics};}

test('7차 계획은 21조건·2데이터셋·2과제이며 512/1024 조건을 섞지 않습니다',()=>{
 assert.equal(CONDITIONS.length,21);assert.deepEqual(LEVELS,[0,1,2,4,8,16,32,64,128,256]);
 const rows=adaptConditions(null,null,null,false);assert.equal(rows.length,84);assert.equal(new Set(rows.map(r=>[r.dataset,r.task,r.condition].join('/'))).size,84);assert.ok(rows.every(r=>r.status==='waiting'&&r.metrics===null));
 assert.equal(previewName('original_000'),'00_original.png');assert.equal(previewName('blur_256'),'10_blur_256.png');assert.equal(previewName('mosaic_256'),'20_mosaic_256.png');assert.equal(previewName('blur_512'),null);
});

test('현재 포인터의 path는 무시하고 검증한 UUID의 7차 기록만 연결합니다',()=>fixture(async f=>{
 await f.write('progress.json',training);const s=await(await f.get('/api/status')).json();assert.equal(s.revision,10);assert.equal(s.experiment.execution_revision,7);assert.equal(s.read_only,true);assert.equal(s.run_id,ID);assert.equal(s.progress.global_step,57);assert.equal(s.process_verified,false);assert.equal(s.conditions.length,84);assert.equal(s.completed_task_conditions,0);
}));

test('잘못된 UUID·누락 포인터는 다른 차수나 임의 경로로 대체하지 않습니다',()=>fixture(async f=>{
 for(const run_id of ['../../5차','smoke-test','12345678/../../other']){
  await fs.writeFile(path.join(f.work,'current_run.json'),JSON.stringify({run_id,path:f.run}));const s=await readSevenStatus(f.dir);assert.equal(s.pointer_error,true);assert.equal(s.progress,null);assert.equal(s.run_id,null);
 }
 await fs.rm(path.join(f.work,'current_run.json'));assert.equal((await readSevenStatus(f.dir)).pointer_error,true);
}));

test('오류 원값 0과 마지막 정상 합성 처리 배치 57은 별도로 보존하며 실행중으로 표시하지 않습니다',()=>fixture(async f=>{
 const failed={phase:'failed',status:'failed',epoch:0,epochs:0,step:0,loss:null,error:'WinError 5',updated_at:'2030-01-02T10:00:15Z'};
 await f.write('progress.json',failed);await f.write('events.jsonl',JSON.stringify(training)+'\n'+JSON.stringify(failed)+'\n');
 const s=await readSevenStatus(f.dir);assert.deepEqual(s.progress,failed);assert.deepEqual(s.last_observed_progress,training);assert.equal(JSON.parse(s.progress_raw).step,0);assert.equal(trainingDisplay(s).progress.global_step,57);assert.equal(trainingDisplay(s).lastObserved,true);assert.equal(describeExperiment(s).status,'실험 오류');assert.ok(trainingPhaseStates(s).every(p=>!p.active));
}));

test('오래된 training과 최근 training 모두 프로세스 미확인을 명시합니다',()=>{
 assert.match(describeExperiment({progress:training,fresh:false}).status,/이전.*미확인/);assert.match(describeExperiment({progress:training,fresh:true}).status,/최근.*미확인/);
 assert.ok(trainingPhaseStates({progress:training,fresh:true}).every(p=>!p.active));
});

test('Host와 쓰기 요청을 거부하고 원래 실행·중단 파일을 생성하지 않습니다',()=>fixture(async f=>{
 const before=await fs.readdir(f.work);
 const bad=await new Promise((resolve,reject)=>{const req=http.request(f.base+'/api/status',{headers:{Host:'outside.invalid'}},res=>{res.resume();resolve(res.statusCode);});req.on('error',reject);req.end();});assert.equal(bad,403);
 for(const url of ['/api/start','/api/stop','/api/resume','/api/status'])for(const method of ['POST','PUT','DELETE'])assert.equal((await fetch(f.base+url,{method,headers:{Origin:f.base}})).status,405);
 assert.deepEqual(await fs.readdir(f.work),before);assert.deepEqual(await fs.readdir(f.run),[]);
}));

test('42 CSV행은84 과제상태이며 한 과제 미완료를 전체 완료로 만들지 않습니다',()=>fixture(async f=>{
 const report={schema:1,completed:true,results:{NIH:{},COVID19:{}}};for(const ds of ['NIH','COVID19'])for(const condition of CONDITIONS)report.results[ds][condition.id]=reportRow(condition);
 report.results.NIH.blur_001.retrieval_status='pending';await f.write('privacy_results.json',report);
 let s=await readSevenStatus(f.dir);assert.equal(s.completed_task_conditions,83);assert.notEqual(describeExperiment(s).status,'84개 과제·조건 결과 기록');
 report.results.NIH.blur_001.retrieval_status='completed';await f.write('privacy_results.json',report);s=await readSevenStatus(f.dir);assert.equal(s.completed_task_conditions,84);assert.equal(describeExperiment(s).status,'84개 과제·조건 결과 기록');
}));

test('CSV 전체 십진 문자열·0·빈값을 보존하고 상태와 측정값 둘 다 확인합니다',()=>fixture(async f=>{
 const csv='dataset,condition,mode,strength,verification_status,verification_auc,verification_accuracy,verification_n_pairs,retrieval_status,retrieval_precision_at_1,retrieval_map_at_r,retrieval_r_precision,retrieval_eligible_queries,privacy_score_at_1\nNIH,original_000,original,0,completed,0.994012345678901234,0.987899999999999999,100,pending,,,,,\nCOVID19,original_000,original,0,completed,0,0,10,completed,0,0,0,5,100\n';
 await f.write('privacy_results.csv',csv);const s=await readSevenStatus(f.dir),v=s.conditions.find(r=>r.dataset==='NIH'&&r.task==='verification'&&r.condition==='original_000'),r=s.conditions.find(r=>r.dataset==='NIH'&&r.task==='retrieval'&&r.condition==='original_000');
 assert.equal(v.metrics.auc,'0.994012345678901234');assert.equal(v.raw.verification_accuracy,'0.987899999999999999');assert.equal(r.status,'waiting');assert.equal(r.metrics,null);assert.equal(s.completed_task_conditions,3);assert.equal(await fs.readFile(path.join(f.run,'privacy_results.csv'),'utf8'),csv);
}));

test('중복·조건 불일치·비유한·JSON과CSV 충돌은 측정값으로 채택하지 않습니다',()=>{
 const base={dataset:'NIH',condition:'original_000',mode:'original',strength:'0',verification_status:'completed',verification_auc:'0.9',verification_accuracy:'0.8'};
 for(const csv of [[base,base],[{...base,mode:'blur'}],[{...base,verification_auc:'NaN'}]]){
  const rows=adaptConditions(null,csv,null,false);const row=rows.find(r=>r.dataset==='NIH'&&r.task==='verification'&&r.condition==='original_000');assert.equal(row.status,'error');assert.equal(row.metrics,null);
 }
 const report={results:{NIH:{original_000:reportRow()}}};const row=adaptConditions(report,[base],null,false).find(r=>r.dataset==='NIH'&&r.task==='verification'&&r.condition==='original_000');assert.equal(row.status,'error');
});

test('대표사진은 현재 UUID·고정 데이터셋·조건 이름만 허용합니다',()=>fixture(async f=>{
 const folder=path.join(f.run,'대표사진/NIH');await fs.mkdir(folder,{recursive:true});const image=Buffer.from('fixture-png');await fs.writeFile(path.join(folder,'00_original.png'),image);
 const response=await f.get('/api/preview/NIH/original_000.png');assert.equal(response.status,200);assert.deepEqual(Buffer.from(await response.arrayBuffer()),image);
 for(const url of ['/api/preview/NIH/blur_512.png','/api/preview/NIH/original_000.png?path=../../progress.json','/api/preview/COVID19/original_000.png','/api/preview/NIH/%2e%2e%2fprogress.json','/server.mjs','/adapter.mjs'])assert.equal((await f.get(url)).status,404,url);
}));

test('연결 경계와 파일크기 상한을 검사하며 실제 링크를 생성하지 않습니다',()=>fixture(async f=>{
 await f.write('too-big.json','x'.repeat(2049));await assert.rejects(boundedFile(f.run,'too-big.json',2048));
 const original=fs.realpath;fs.realpath=async file=>String(file).endsWith('too-big.json')?path.join(f.dir,'outside.json'):original(file);
 try{await assert.rejects(boundedFile(f.run,'too-big.json'));}finally{fs.realpath=original;}
}));

test('JSONL은256KiB와 LF경계를 지키며 손상·미완성 기록을 명시합니다',()=>fixture(async f=>{
 const file=path.join(f.run,'events.jsonl');await fs.writeFile(file,'가'.repeat(100000)+'\n'+JSON.stringify({...training,note:'앞\u0085뒤'})+'\n{"partial":');
 let value=await readTrainingHistory(file,ID);assert.equal(value.truncated,true);assert.equal(value.incomplete_tail,true);assert.equal(value.last_observed_progress.global_step,57);
 await fs.writeFile(file,JSON.stringify(training)+'\n{broken}\n');value=await readTrainingHistory(file,ID);assert.equal(value.status,'malformed');assert.equal(value.last_observed_progress,null);
}));

test('정확한 십진 반올림·100%구분·원값·미측정 정책을 유지합니다',()=>{
 assert.equal(formatPercent('0.9878048780487805'),'98.8%');assert.equal(formatPercent('0.987499999999999999'),'98.7%');assert.equal(formatPercent('9.875e-1'),'98.8%');assert.equal(formatPercent('98.789',{ratio:false}),'98.8%');assert.match(percentNotice('0.9999999999999999999'),/100% 미만/);assert.match(percentNotice('1'),/정확히 1/);assert.match(percentNotice('100',{ratio:false}),/정확히 100/);assert.equal(formatRaw(0),'0');for(const value of [null,'',NaN,Infinity])assert.equal(formatRaw(value),'미측정');
});

test('정적화면은10차·실험7차이며 테마·모션·키보드선택·원값펼침을 유지합니다',()=>fixture(async f=>{
 const response=await f.get('/');assert.equal(response.status,200);const html=await response.text(),js=await fs.readFile(path.join(ROOT,'화면/app.js'),'utf8'),css=await fs.readFile(path.join(ROOT,'화면/style.css'),'utf8');
 assert.match(html,/화면 10차 · 실험 7차/);assert.match(html,/84개 상태/);assert.doesNotMatch(html,/id="(?:start|stop)"/);assert.doesNotMatch(js,/method:'POST'|innerHTML/);assert.match(js,/history.replaceState/);assert.match(js,/data-dataset/);assert.match(js,/ArrowLeft/);assert.match(js,/expanded.has/);assert.match(js,/localStorage.getItem\('cxr-theme'\)/);assert.match(js,/timeZone:'Asia\/Seoul'/);
 assert.match(css,/\.motion:hover,\.motion:active,\.motion:disabled\{background:transparent\}/);assert.match(css,/min-height:44px/);assert.match(css,/prefers-reduced-motion/);assert.match(css,/data-theme=dark/);assert.match(response.headers.get('content-security-policy'),/default-src 'self'/);
}));

test('현재 UUID의 가장 최근 stdout/stderr 쌍을 읽고 오래된 고정 로그와 섞지 않습니다',()=>fixture(async f=>{
 await f.write('training.stdout.log','old fixed');await f.write('training.stdout.log.20300101-010000-abcdef.stdout.log','old redirected');await f.write('training.stdout.log.20300102-020000-123abc.stdout.log','new output');await f.write('training.stdout.log.20300102-020000-123abc.stderr.log','new warning');await f.write('training.stdout.log.99999999-999999-escape.stderr.log','bad name');
 const s=await readSevenStatus(f.dir);assert.match(s.logs.training.text,/new output/);assert.match(s.logs.training.text,/new warning/);assert.doesNotMatch(s.logs.training.text,/old fixed|old redirected|bad name/);assert.equal(s.logs.training.sources.length,2);
}));

test('프로세스 신원은 정확한 source·UUID출력·부모·생성시각으로 redirector를 한 실행으로 묶습니다',async()=>{
 const {matchProcesses}=await import('../process.mjs');const root='E:/실험/7차',now=Date.parse('2030-01-02T04:00:00Z');
 const manager={pid:1,ppid:10,created_at:'2030-01-02T03:49:00Z',command_line:`powershell.exe -File "${root}/All-In-One.ps1" -Action resume -RunID ${ID}`};
 const argv=`-u "${root}/src/train_from_scratch.py" --project "E:/실험/연구 자료" --output "${root}/runs/${ID}" --resume`;
 const redirect={pid:2,ppid:1,created_at:'2030-01-02T03:50:00Z',command_line:'venvpython.exe '+argv},child={pid:3,ppid:2,created_at:'2030-01-02T03:50:01Z',command_line:'basepython.exe '+argv};
 const match=matchProcesses([manager,redirect,child],root,ID,'2030-01-01T00:00:00Z',now);assert.equal(match.verified,true);assert.equal(match.leaf_count,1);assert.equal(match.proofs[0].pid,3);assert.deepEqual(match.proofs[0].redirector_pids,[2]);
 for(const change of [{...child,command_line:child.command_line.replace(ID,ID+'-outside')},{...child,ppid:99},{...child,created_at:'2030-01-02T03:48:00Z'},{...child,command_line:child.command_line.replace('train_from_scratch.py','other.py')}]){
  const altered=matchProcesses([manager,change],root,ID,'2030-01-01T00:00:00Z',now);assert.equal(altered.verified,false);
 }
 const wrongManager={...manager,command_line:manager.command_line.replace(ID,'other')};assert.equal(matchProcesses([wrongManager,redirect,child],root,ID,null,now).verified,false);
 const duplicate={...child,pid:4,ppid:1};assert.equal(matchProcesses([manager,redirect,child,duplicate],root,ID,null,now).status,'ambiguous');
});

test('신선한학습기록과 정확한프로세스 양쪽이 있어야 학습중이며 이전오류는 보존합니다',()=>{
 const process={verified:true,proofs:[{script:'train_from_scratch.py',pid:3}]};
 assert.match(describeExperiment({progress:training,fresh:true,process}).status,/모델 학습 중/);assert.equal(trainingPhaseStates({progress:training,fresh:true,process})[0].active,true);assert.doesNotMatch(describeExperiment({progress:training,fresh:false,process}).status,/학습 중/);
 const failed={...training,status:'failed',phase:'failed'};assert.match(describeExperiment({progress:failed,fresh:true,process}).status,/재개 프로세스 확인.*이전 오류/);assert.equal(failed.status,'failed');
});

test('프로세스조회실패는 확인불가이며10초캐시로 과다 CIM호출을 막습니다',async()=>{
 const {createProcessObserver}=await import('../process.mjs');let calls=0;const observer=createProcessObserver(async()=>{calls++;throw Error('fixture failure');});
 const results=await Promise.all([observer('E:/실험/7차',ID,null),observer('E:/실험/7차',ID,null)]);assert.equal(calls,1);assert.ok(results.every(result=>result.status==='unavailable'&&!result.verified));await observer('E:/실험/7차',ID,null);assert.equal(calls,1);
});

test('잘못된 UTF-8 JSON·CSV는 대체문자로 채택하지 않고 읽기 오류로 드러냅니다',()=>fixture(async f=>{
 await fs.writeFile(path.join(f.run,'privacy_results.csv'),Buffer.from([100,97,116,97,115,101,116,10,0xff]));await fs.writeFile(path.join(f.run,'progress.json'),Buffer.from([123,34,0xff,34,58,49,125]));
 const s=await readSevenStatus(f.dir);assert.equal(s.progress,null);assert.equal(s.completed_task_conditions,0);assert.ok(s.errors.some(e=>e.source==='privacy_results.csv'));assert.ok(s.errors.some(e=>e.source==='progress.json'));
}));

test('원본 성능은 같은 실행 모델 SHA가 일치할 때만 먼저 표시하며 변환 완료로 확장하지 않습니다',()=>{
 const sha='a'.repeat(64),artifacts={published_cxr_initialization:false,verification_sha256:sha},baseline={NIH:{verification:metrics.verification,verification_sha256:sha}};
 let rows=adaptConditions(null,null,null,false,baseline,artifacts);assert.equal(rows.filter(r=>r.status==='complete').length,1);assert.equal(rows.find(r=>r.status==='complete').source,'evaluation_results.json');assert.equal(rows.find(r=>r.dataset==='NIH'&&r.task==='verification'&&r.condition==='blur_000').status,'waiting');
 rows=adaptConditions(null,null,null,false,baseline,{...artifacts,verification_sha256:'b'.repeat(64)});assert.equal(rows.filter(r=>r.status==='complete').length,0);assert.equal(rows.find(r=>r.dataset==='NIH'&&r.task==='verification'&&r.condition==='original_000').status,'error');
});

test('정상 이력 생략은 한국어 보조 안내로 표시하고 손상·읽기 실패만 오류로 표시합니다',async()=>{
 const {historyNotice}=await import('../화면/state.mjs');
 const truncated=historyNotice({status:'truncated',truncated:true});assert.equal(truncated.error,'');assert.match(truncated.info,/최근 256 KiB/);assert.match(truncated.info,/앞부분.*생략/);assert.doesNotMatch(truncated.info,/truncated/);
 const partial=historyNotice({status:'incomplete_tail',truncated:true,incomplete_tail:true});assert.equal(partial.error,'');assert.match(partial.info,/작성 중인 마지막 행/);assert.match(partial.info,/앞부분.*생략/);
 for(const status of ['malformed','unreadable','invalid_record','different_run','invalid_training_record','unknown']){
  const result=historyNotice({status,truncated:true});assert.ok(result.error.length>0);assert.doesNotMatch(result.error,new RegExp(status));assert.match(result.info,/원본 파일은 보존/);
 }
 assert.deepEqual(historyNotice(null),{info:'',error:''});assert.equal(historyNotice({status:'ok'}).error,'');assert.equal(historyNotice({status:'missing'}).error,'');
 const js=await fs.readFile(path.join(ROOT,'화면/app.js'),'utf8'),html=await fs.readFile(path.join(ROOT,'화면/index.html'),'utf8');assert.match(js,/historyMessage.error/);assert.match(js,/history-note/);assert.match(html,/id="history-note"/);assert.doesNotMatch(js,/'events.jsonl: '\+s.history.status/);
});

test('공격 실패율과 처리 배치 수는 가중치 적용 횟수·보수 표현과 구분합니다',async()=>{
 const js=await fs.readFile(path.join(ROOT,'화면/app.js'),'utf8');assert.match(js,/고정 공격의 1순위 재식별 실패율이며 익명성 보장이 아닙니다/);assert.doesNotMatch(js,/공격 실패율의 보수/);assert.match(js,/단계 내 누적 처리 배치/);assert.match(js,/처리 배치 수는 가중치 적용 횟수가 아닙니다/);assert.doesNotMatch(js,/누적 갱신 \$\{metric\(training.global_step\)/);
});

test('현재 에폭과 최대 예정 단계는 정확한 분모만 사용하고 미지값을0으로 채우지 않습니다',async()=>{
 const {trainingFractions}=await import('../화면/state.mjs');
 const result=trainingFractions(training);assert.equal(result.epoch,17/40);assert.equal(result.phase,(1+17/40)/9);assert.equal(result.remaining,23);assert.equal(formatPercent(result.epoch),'42.5%');assert.equal(formatPercent(result.phase),'15.8%');
 assert.equal(trainingFractions({...training,step:0}).epoch,0);
 for(const value of [null,{...training,steps:0},{...training,steps:null},{...training,step:41},{...training,epoch:0},{...training,epoch:10},{...training,epoch:1.5}])assert.equal(trainingFractions(value),null);
});

test('실제 학습·에폭 검증 기록을 반환하고 손상행과 다른실행이면 전체 시계열을 채택하지 않습니다',()=>fixture(async f=>{
 const validation={phase:'verification',status:'epoch_complete',epoch:2,epochs:9,step:40,loss:0.1,validation_loss:0.2,updated_at:'2030-01-02T10:01:00Z'};
 await f.write('events.jsonl',JSON.stringify(training)+'\n'+JSON.stringify(validation)+'\n');let h=await readTrainingHistory(path.join(f.run,'events.jsonl'),ID);assert.deepEqual(h.loss_samples,[training]);assert.deepEqual(h.validation_samples,[validation]);
 const s=await readSevenStatus(f.dir);assert.deepEqual(s.history.validation_samples,[validation]);
 for(const bad of ['{broken}',JSON.stringify({...validation,validation_loss:null}),JSON.stringify({...training,run_id:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'})]){
  await f.write('events.jsonl',JSON.stringify(training)+'\n'+bad+'\n');h=await readTrainingHistory(path.join(f.run,'events.jsonl'),ID);assert.deepEqual(h.loss_samples,[]);assert.deepEqual(h.validation_samples,[]);assert.equal(h.last_observed_progress,null);
 }
}));

test('손실 선은 단계·에폭·재개 되감기·시간 역전 경계에서 분리하고0손실은 유지합니다',async()=>{
 const {lossSegments}=await import('../화면/state.mjs');
 const point=(step,time,other={})=>({...training,global_step:step,updated_at:`2030-01-02T10:00:${String(time).padStart(2,'0')}Z`,...other});
 const segments=lossSegments([point(1,1,{loss:0}),point(2,2),point(1,3),point(2,4,{epoch:3}),point(3,5,{epoch:3,phase:'retrieval_phase1'}),point(4,4,{epoch:3,phase:'retrieval_phase1'})]);
 assert.deepEqual(segments.map(x=>x.length),[2,1,1,1,1]);assert.equal(segments[0][0].value,0);assert.deepEqual(lossSegments([{...training,loss:null}]),[]);
 assert.equal(lossSegments([point(1,1,{epoch:1,validation_loss:0.2}),point(2,2,{epoch:2,validation_loss:0.1})],'validation_loss')[0].length,2);
});

test('단계완료는 가중치존재만으로 주장하지 않고 후속단계 또는 실제전체종료와 결합합니다',async()=>{
 const {phaseTimeline,trainingFractions}=await import('../화면/state.mjs');
 const phases={verification:{selected_weights:true},retrieval_phase1:{selected_weights:true}};
 assert.equal(phaseTimeline({progress:training,training_phases:phases,config:{tasks:'both'}})[0].complete,false);
 const s={progress:{...training,phase:'retrieval_phase1',epoch:1,epochs:30},training_phases:phases,config:{tasks:'both'}};
 assert.equal(phaseTimeline(s)[0].complete,true);assert.equal(phaseTimeline(s)[1].complete,false);assert.equal(phaseTimeline({...s,config:{tasks:'retrieval'}})[0].complete,false);
 const completed=phaseTimeline({...s,progress:{phase:'complete',status:'completed'}});assert.equal(completed[0].complete,true);assert.equal(completed[2].complete,false);
 assert.equal(trainingFractions({...training,epoch:3,epochs:100,step:40}).phase,0.03);
});

test('학습시각화는모바일테마·원값·분모·기록범위를보존하고ETA를만들지않습니다',async()=>{
 const html=await fs.readFile(path.join(ROOT,'화면/index.html'),'utf8'),js=await fs.readFile(path.join(ROOT,'화면/app.js'),'utf8'),css=await fs.readFile(path.join(ROOT,'화면/style.css'),'utf8');
 for(const id of ['epoch-progress','phase-progress','loss-chart','validation-chart'])assert.match(html,new RegExp('id="'+id+'"'));
 assert.match(html,/전체 실험 시간 완료율이 아닙니다/);assert.match(html,/조기 종료/);assert.match(html,/남은 시간은 계산하지 않습니다/);assert.match(js,/progress_fractions:f/);assert.match(js,/lossSegments\(selected,key\)/);assert.match(js,/최근 256 KiB 범위/);assert.match(css,/@media\(max-width:500px\)/);assert.match(css,/stroke:var\(--accent\)/);
});

test('원정수 진행률은 이진 나눗셈 반올림 경계에서도 정확하고 원비율은 보존합니다',async()=>{
 const {formatProgressPercent,trainingFractions}=await import('../화면/state.mjs');
 const a={epoch:1,epochs:100,step:8666,steps:24760},b={...a,epoch:2,step:11142};
 assert.equal(formatProgressPercent(a,'phase'),'0.4%');assert.equal(formatProgressPercent(b,'phase'),'1.5%');
 assert.equal(trainingFractions(a).phase,((a.epoch-1)+a.step/a.steps)/a.epochs);assert.equal(formatProgressPercent({...a,step:0}),'0.0%');assert.equal(formatProgressPercent({...a,step:24760}),'100.0%');
 assert.equal(formatProgressPercent({...a,steps:null}),'미측정');assert.equal(formatProgressPercent(a,'other'),'미측정');
});

test('2100개 에폭·최대예정 경계조건에서 원정수와 독립 구성한 기대 표시를 확인합니다',async()=>{
 const {formatProgressPercent}=await import('../화면/state.mjs');let cases=0;
 const display=tenths=>Math.floor(tenths/10)+'.'+tenths%10+'%';
 for(let epoch=1;epoch<=100;epoch++)for(let index=0;index<=20;index++){
  // 단계 백분율은 (epoch−1)+index/20 이므로 홀수 index가 정확히 반올림 경계입니다.
  const p={epoch,epochs:100,step:index*100,steps:2000};
  assert.equal(formatProgressPercent(p,'phase'),display((epoch-1)*10+Math.floor((index+1)/2)));
  assert.equal(formatProgressPercent(p,'epoch'),display(index*50));
  // 에폭 백분율은 index/10+0.05 이므로 항상 올림 경계입니다.
  assert.equal(formatProgressPercent({...p,step:index*2+1},'epoch'),display(index+1));cases++;
 }
 assert.equal(cases,2100);
});

async function independentFixture(f,change=()=>{}){
 const {INDEPENDENT_CONDITIONS}=await import('../화면/state.mjs');const {independentFingerprint,INDEPENDENT_PROTOCOL}=await import('../adapter.mjs');
 const state={schema_version:1,protocol:INDEPENDENT_PROTOCOL,queue_id:'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',status:'waiting_baseline',updated_at:'2030-02-01T00:00:00Z',current_condition_id:null,conditions:INDEPENDENT_CONDITIONS.map((p,i)=>({id:p.id,mode:p.mode,strength:p.strength,identity_transform:p.identity,status:'pending',run_id:null,training_subdir:null,condition_fingerprint:independentFingerprint(p.id),initialization:'fresh_imagenet_seed3',error:null}))};
 change(state);await fs.mkdir(path.join(f.work,'queue'),{recursive:true});await fs.writeFile(path.join(f.work,'queue/state.json'),JSON.stringify(state));return state;
}
test('독립학습20조건은원본21조건평가와분리되며항등조건도별도실행으로대기합니다',()=>fixture(async f=>{
 await independentFixture(f);const s=await(await f.get('/api/status')).json();assert.equal(s.revision,10);assert.equal(s.conditions.length,84);assert.equal(s.independent.conditions.length,20);assert.equal(s.independent.conditions.filter(c=>c.identity).length,3);assert.ok(s.independent.conditions.every(c=>c.status==='waiting_original'));
 const selected=await(await f.get('/api/independent/blur_000')).json();assert.equal(selected.condition.id,'blur_000');assert.equal(selected.status,'waiting_original');assert.equal(selected.progress,null);assert.equal(selected.history,null);assert.equal(s.completed_task_conditions,0);
}));
test('독립조건ID는allowlist만허용하고쿼리·경로·실제실행UUID를입력으로받지않습니다',()=>fixture(async f=>{
 await independentFixture(f);
 for(const url of ['/api/independent/original_000','/api/independent/blur_512','/api/independent/'+ID,'/api/independent/blur_000?path=../../progress.json','/api/independent/%2e%2e%2fprogress.json'])assert.equal((await f.get(url)).status,404,url);
 assert.equal((await fetch(f.base+'/api/independent/blur_000',{method:'POST'})).status,405);
}));
test('누락·중복조건·임의폴더·동일run공유·mode충돌큐는기록을채택하지않습니다',()=>fixture(async f=>{
 const {readIndependentOverview}=await import('../adapter.mjs');assert.equal((await readIndependentOverview(f.dir)).status,'unknown');
 for(const change of [s=>s.conditions.pop(),s=>s.conditions[1]={...s.conditions[0]},s=>s.conditions[0].mode='mosaic',s=>{s.conditions[0].run_id=ID;s.conditions[0].training_subdir='../../runs/'+ID},s=>{for(const row of s.conditions.slice(0,2)){row.run_id=ID;row.training_subdir='work/runs/'+ID+'/training'}}]){
  await independentFixture(f,change);const result=await readIndependentOverview(f.dir);assert.equal(result.status,'unknown');assert.ok(result.errors.length);assert.ok(result.conditions.every(c=>c.status==='unknown'));
 }
}));
test('원본학습기록과다른독립조건이선택조건의빈기록을채우지않습니다',()=>fixture(async f=>{
 await f.write('progress.json',training);await f.write('events.jsonl',JSON.stringify(training)+'\n');await independentFixture(f);
 const s=await(await f.get('/api/status')).json();assert.equal(s.progress.global_step,57);
 for(const id of ['blur_000','mosaic_001']){const value=await(await f.get('/api/independent/'+id)).json();assert.equal(value.condition.id,id);assert.equal(value.progress,null);assert.equal(value.history,null);assert.equal(value.initialization,null);}
}));
test('독립조건선택은즉시화면을비우고늦은다른조건응답을거부하며원본진행을교체하지않습니다',async()=>{
 const js=await fs.readFile(path.join(ROOT,'화면/app.js'),'utf8'),html=await fs.readFile(path.join(ROOT,'화면/index.html'),'utf8');
 assert.match(js,/independentSelection=id;renderIndependentDetail\(null\)/);assert.match(js,/id===independentSelection&&request===independentRequest&&value.condition\?\.id===id/);assert.match(js,/drawLoss\('independent-loss-chart'/);assert.match(html,/원본 7차 학습/);assert.match(html,/고정 모델의 21조건 평가/);assert.match(html,/강도별 독립 학습/);assert.match(html,/항등 변환도 독립 실행/);assert.match(js,/원본이나 다른 조건의 기록을 대신 표시하지 않습니다/);
});

async function independentTrainingFixture(f,id='blur_000'){
 const state=await independentFixture(f,s=>{s.status='training';s.current_condition_id=id;const row=s.conditions.find(c=>c.id===id);row.status='training';row.run_id=ID;row.training_subdir='work/runs/'+ID+'/training';});
 const row=state.conditions.find(c=>c.id===id),folder=path.join(f.dir,row.training_subdir);await fs.mkdir(folder,{recursive:true});
 const {independentDefinition,INDEPENDENT_PROTOCOL}=await import('../adapter.mjs');
 const config={tasks:'both',condition_id:id,condition_fingerprint:row.condition_fingerprint,run_id:ID,condition:independentDefinition(id),initialization_mode:'fresh_imagenet_seed3',cross_condition_state_inherited:false,experiment_protocol:INDEPENDENT_PROTOCOL};
 await fs.writeFile(path.join(folder,'training_config.json'),JSON.stringify(config));await fs.writeFile(path.join(folder,'progress.json'),JSON.stringify(training));await fs.writeFile(path.join(folder,'events.jsonl'),JSON.stringify({...training,run_id:ID})+'\n');return {folder,config,row};
}
test('조건config·초기화지문불일치면다른조건의진행과손실을표시하지않습니다',()=>fixture(async f=>{
 const {readIndependentCondition}=await import('../adapter.mjs');const {folder,config}=await independentTrainingFixture(f);
 let value=await readIndependentCondition(f.dir,'blur_000');assert.equal(value.progress.global_step,57);assert.equal(value.history.loss_samples.length,1);assert.equal(value.status,'stale');
 for(const changed of [{...config,condition_id:'mosaic_000'},{...config,run_id:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'},{...config,condition_fingerprint:'f'.repeat(64)},{...config,condition:{...config.condition,strength:1}}]){
  await fs.writeFile(path.join(folder,'training_config.json'),JSON.stringify(changed));value=await readIndependentCondition(f.dir,'blur_000');assert.equal(value.progress,null);assert.equal(value.history,null);assert.ok(value.errors.length);
 }
 await fs.writeFile(path.join(folder,'training_config.json'),JSON.stringify(config));await fs.mkdir(path.join(folder,'verification'));await fs.writeFile(path.join(folder,'verification/initialization.json'),JSON.stringify({...config,condition_id:'mosaic_000'}));value=await readIndependentCondition(f.dir,'blur_000');assert.equal(value.progress,null);assert.equal(value.history,null);
}));
test('선택조건은다른조건및원본history와분리하고대기조건에이전곡선을넘기지않습니다',()=>fixture(async f=>{
 await independentTrainingFixture(f);await f.write('progress.json',{...training,global_step:900});
 const a=await(await f.get('/api/independent/blur_000')).json(),b=await(await f.get('/api/independent/blur_001')).json(),s=await(await f.get('/api/status')).json();
 assert.equal(a.progress.global_step,57);assert.equal(a.history.loss_samples[0].global_step,57);assert.equal(b.progress,null);assert.equal(b.history,null);assert.equal(s.progress.global_step,900);assert.equal(a.process.verified,false);assert.notEqual(a.status,'training');
}));
test('독립훈련프로세스는실제PID생성시각·조건·출력·queue계보가모두맞아야확인합니다',async()=>{
 const {matchIndependentProcesses}=await import('../process.mjs');const root='E:/합성/10차',id='blur_000',run=ID,fingerprint='a'.repeat(64),created='2030-01-02T03:50:00Z',now=Date.parse('2030-01-02T04:00:00Z');
 const argv=`-u "${root}/src/train_from_scratch.py" --condition-id ${id} --output "${root}/work/runs/${run}/training" --project "E:/합성/연구 자료"`;
 const parent={pid:10,ppid:1,created_at:'2030-01-02T03:49:00Z',command_line:`python.exe "${root}/src/cxr10_queue.py"`},redirect={pid:20,ppid:10,created_at:created,command_line:'venv.exe '+argv},child={pid:30,ppid:20,created_at:'2030-01-02T03:50:01Z',command_line:'base.exe '+argv};
 const claim={pid:20,creation_filetime:String(BigInt(Date.parse(created))*10000n+116444736000000000n),entry:'train_from_scratch.py',command_line:redirect.command_line,run_id:run,condition_id:id,condition_fingerprint:fingerprint},condition={id,run_id:run,fingerprint};
 const good=matchIndependentProcesses([parent,redirect,child],root,claim,condition,now);assert.equal(good.verified,true);assert.equal(good.proofs[0].pid,30);assert.deepEqual(good.proofs[0].redirector_pids,[20]);
 for(const altered of [{...claim,creation_filetime:'1'},{...claim,condition_id:'blur_001'},{...claim,condition_fingerprint:'b'.repeat(64)},{...claim,command_line:claim.command_line.replace('/training','/training-other')}])assert.equal(matchIndependentProcesses([parent,redirect,child],root,altered,condition,now).verified,false);
 assert.equal(matchIndependentProcesses([{...parent,command_line:'unrelated.exe'},redirect,child],root,claim,condition,now).verified,false);
 assert.equal(matchIndependentProcesses([parent,redirect,child,{...child,pid:40}],root,claim,condition,now).verified,false);
});
test('동시조건실행및대기큐의학습주장은검증오류이며새학습으로표시하지않습니다',()=>fixture(async f=>{
 const {readIndependentOverview}=await import('../adapter.mjs');await independentFixture(f,s=>{s.conditions[0].status='training';s.conditions[0].run_id=ID;s.conditions[0].training_subdir='work/runs/'+ID+'/training';});const value=await readIndependentOverview(f.dir);assert.equal(value.status,'unknown');assert.ok(value.errors.length);
}));

test('독립완료는큐문구만믿지않고현재조건의선택가중치와NIH/COVID양과제결과결합을확인합니다',()=>fixture(async f=>{
 const {readIndependentCondition}=await import('../adapter.mjs');const {folder,config}=await independentTrainingFixture(f);
 const queue=JSON.parse(await fs.readFile(path.join(f.work,'queue/state.json'),'utf8'));queue.conditions[0].status='completed';await fs.writeFile(path.join(f.work,'queue/state.json'),JSON.stringify(queue));
 assert.equal((await readIndependentCondition(f.dir,'blur_000')).status,'unknown');
 const sha='a'.repeat(64),artifacts={...config,published_cxr_initialization:false,verification_sha256:sha,retrieval_sha256:sha},evaluation={...config,completed:true};
 for(const ds of ['NIH','COVID19'])evaluation[ds]={verification:metrics.verification,retrieval:metrics.retrieval,verification_sha256:sha,retrieval_sha256:sha};
 for(const phase of ['verification','retrieval_phase2']){await fs.mkdir(path.join(folder,phase),{recursive:true});await fs.writeFile(path.join(folder,phase,'selected_weights.pth'),'synthetic-checkpoint');}
 await fs.writeFile(path.join(folder,'artifacts.json'),JSON.stringify(artifacts));await fs.writeFile(path.join(folder,'evaluation_results.json'),JSON.stringify(evaluation));
 assert.equal((await readIndependentCondition(f.dir,'blur_000')).status,'complete');
 evaluation.COVID19.retrieval_sha256='b'.repeat(64);await fs.writeFile(path.join(folder,'evaluation_results.json'),JSON.stringify(evaluation));assert.equal((await readIndependentCondition(f.dir,'blur_000')).status,'unknown');
 evaluation.COVID19.retrieval_sha256=sha;evaluation.condition={...config.condition,strength:4};await fs.writeFile(path.join(folder,'evaluation_results.json'),JSON.stringify(evaluation));assert.equal((await readIndependentCondition(f.dir,'blur_000')).progress,null);
}));


test('새21주탐색의강도별학습은실제영역으로이동하고이전평가와분리합니다',async()=>{
 const html=await fs.readFile(path.join(ROOT,'화면/index.html'),'utf8'),js=await fs.readFile(path.join(ROOT,'화면/app.js'),'utf8');
 const nav=html.match(/<nav[^>]*>([\s\S]*?)<\/nav>/)[1];
 assert.match(nav,/<a href="#independent-training"[^>]*>강도별 학습<\/a>/);assert.match(html,/<section[^>]*id="independent-training"/);
 assert.match(html,/10차 새 원본 \+ 블러10 \+ 모자이크10/);assert.match(html,/새 원본 10차 학습/);assert.match(html,/원본1 \+ 독립20 = 21개 학습 조건/);
 assert.match(html,/id="original-detail"/);assert.match(js,/selectIndependent\('original_000'\)/);assert.match(js,/TRAINING_CONDITIONS\.find/);
 assert.match(js,/document\.querySelectorAll\('nav a'\)/);assert.match(js,/a\.classList\.add\('selected'\)/);
});

test('새21조건은항등원본정의와별도지문을가지며20조건선택을보존합니다',()=>fixture(async f=>{
 const {TRAINING_CONDITIONS,INDEPENDENT_CONDITIONS}=await import('../화면/state.mjs');
 const {independentDefinition,independentFingerprint,readIndependentOverview,readIndependentCondition}=await import('../adapter.mjs');
 await independentFixture(f);assert.equal((await readIndependentCondition(f.dir,'original_000')).progress,null);
 assert.equal(TRAINING_CONDITIONS.length,21);assert.equal(INDEPENDENT_CONDITIONS.length,20);
 assert.deepEqual(independentDefinition('original_000'),{id:'original_000',mode:'original',strength:0,identity_transform:true,operator_mode:'original',operator_strength:0,units:'none',training_sides:'all_training_and_validation_images',evaluation_sides:'transformed_query_and_transformed_gallery',original_image_modified:false});
 assert.notEqual(independentFingerprint('original_000'),independentFingerprint('blur_000'));
 await independentFixture(f,s=>{s.start_mode='fresh_all_21';s.baseline={status:'not_required'};s.status='ready';s.conditions.unshift({id:'original_000',mode:'original',strength:0,identity_transform:true,status:'pending',run_id:null,training_subdir:null,condition_fingerprint:independentFingerprint('original_000'),initialization:'fresh_imagenet_seed3',error:null});});
 const value=await readIndependentOverview(f.dir);assert.equal(value.conditions.length,21);assert.equal(value.conditions[0].id,'original_000');assert.equal(value.conditions[0].status,'pending');
 const state=JSON.parse(await fs.readFile(path.join(f.work,'queue/state.json'),'utf8'));state.conditions[1].run_id=ID;state.conditions[1].training_subdir='work/runs/'+ID+'/training';state.conditions[0].run_id=ID;state.conditions[0].training_subdir='work/runs/'+ID+'/training';await fs.writeFile(path.join(f.work,'queue/state.json'),JSON.stringify(state));assert.equal((await readIndependentOverview(f.dir)).status,'unknown');
}));

test('새21상단은10차원본만읽고7차학습기록을현재값으로혼합하지않습니다',()=>fixture(async f=>{
 const {independentDefinition,independentFingerprint,INDEPENDENT_PROTOCOL}=await import('../adapter.mjs');
 await f.write('progress.json',{...training,global_step:9999});
 await independentFixture(f,s=>{s.start_mode='fresh_all_21';s.baseline={status:'not_required'};s.status='training';s.current_condition_id='original_000';s.conditions.unshift({id:'original_000',mode:'original',strength:0,identity_transform:true,status:'training',run_id:ID,training_subdir:'work/runs/'+ID+'/training',condition_fingerprint:independentFingerprint('original_000'),initialization:'fresh_imagenet_seed3',error:null});});
 const folder=path.join(f.dir,'work/runs',ID,'training');await fs.mkdir(folder,{recursive:true});
 const config={condition_id:'original_000',condition_fingerprint:independentFingerprint('original_000'),condition:independentDefinition('original_000'),run_id:ID,initialization_mode:'fresh_imagenet_seed3',cross_condition_state_inherited:false,experiment_protocol:INDEPENDENT_PROTOCOL};
 await fs.writeFile(path.join(folder,'training_config.json'),JSON.stringify(config));await fs.writeFile(path.join(folder,'progress.json'),JSON.stringify({...training,global_step:3}));await fs.writeFile(path.join(folder,'training.stdout.log'),'synthetic fresh original log');
 const server=createDashboard({root:ROOT,executionRoot:f.dir,independentRoot:f.dir,freshOriginal:true,processQuery:async()=>[],gpuQuery:(_a,_b,_c,cb)=>cb(null,'Test GPU,0,0,24564')});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 try{const s=await(await fetch(`http://127.0.0.1:${server.address().port}/api/status`)).json();assert.equal(s.experiment.execution_revision,10);assert.equal(s.condition.id,'original_000');assert.equal(s.progress.global_step,3);assert.notEqual(s.progress.global_step,9999);assert.equal(s.archived_seven.live_training_source,false);assert.equal(s.process.verified,false);assert.match(s.logs.training.text,/synthetic fresh original/);assert.equal(s.completed_task_conditions,0);}
 finally{await new Promise(resolve=>server.close(resolve));}
}));

test('새21관리자의정규상대경로는절대자식출력과계보에결합하고유사경로는거절합니다',async()=>{
 const {matchIndependentProcesses}=await import('../process.mjs'),root='E:/합성/10차',id='original_000',fingerprint='a'.repeat(64),created='2030-01-02T03:50:00Z',now=Date.parse('2030-01-02T04:00:00Z');
 const command=`python.exe -u "${root}/src/train_from_scratch.py" --condition-id ${id} --output "${root}/work/runs/${ID}/training" --project "E:/합성/연구 자료"`;
 const manager={pid:10,ppid:1,created_at:'2030-01-02T03:49:00Z',command_line:'python.exe -B -X utf8 -u src/cxr10_queue.py --run --poll-seconds 30'},child={pid:20,ppid:10,created_at:created,command_line:command};
 const claim={pid:20,creation_filetime:String(BigInt(Date.parse(created))*10000n+116444736000000000n),entry:'train_from_scratch.py',command_line:command,run_id:ID,condition_id:id,condition_fingerprint:fingerprint},condition={id,run_id:ID,fingerprint};
 const good=matchIndependentProcesses([manager,child],root,claim,condition,now);assert.equal(good.verified,true);assert.equal(good.proofs[0].manager_source,'exact_relative_child_root_bound');assert.equal(good.proofs[0].manager_cwd_observed,false);
 for(const command_line of ['python.exe cxr10_queue.py --run','python.exe ../src/cxr10_queue.py --run','python.exe src/../src/cxr10_queue.py --run','python.exe src/cxr10_queue.py','python.exe src/cxr10_queue.py --run=1','python.exe src/cxr10_queue.py --run --prepare-fresh'])assert.equal(matchIndependentProcesses([{...manager,command_line},child],root,claim,condition,now).verified,false);
 assert.equal(matchIndependentProcesses([manager,{...child,command_line:command.replace('/training','/other')}],root,claim,condition,now).verified,false);
});
