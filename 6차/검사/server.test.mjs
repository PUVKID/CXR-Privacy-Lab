import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import http from 'node:http';
import { createDashboard, parseCSV, tail } from '../server.mjs';
import { describeExperiment } from '../화면/state.mjs';

async function fixture(fn) {
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'cxr-dashboard-'));const research=path.join(dir,'research');const control=path.join(research,'control');await fs.mkdir(control,{recursive:true});
 const children=[];let launches=0;const server=createDashboard({root:dir,research,launch:()=>{launches++;const c=spawn(process.execPath,['-e','setTimeout(()=>{},30000)'],{windowsHide:true});children.push(c);return c;},gpuQuery:(_a,_b,_c,cb)=>cb(null,'Test GPU, 12, 100, 1000')});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
 const request=(url,options={})=>fetch(base+url,options);const post=url=>request(url,{method:'POST',headers:{Origin:base}});
 try{await fn({dir,research,control,request,post,base,launches:()=>launches});}finally{for(const c of children)c.kill();await new Promise(r=>server.close(r));await fs.rm(dir,{recursive:true,force:true});}
}
test('CSV는 BOM·따옴표·개행·누락 값을 보존합니다',()=>{assert.deepEqual(parseCSV('\uFEFFstrength,auc,note\r\n2,0.75,"한글, 값"\r\n4,,"두\n줄"\n'),[{strength:'2',auc:'0.75',note:'한글, 값'},{strength:'4',auc:'',note:'두\n줄'}]);});
test('실제 단계 결과와 부분 다운로드 크기만 반환합니다',()=>fixture(async f=>{
 const folder=path.join(f.research,'results','stage_2_blur');await fs.mkdir(folder,{recursive:true});await fs.writeFile(path.join(folder,'summary.csv'),'stage,strength,auc,cmc_rank_1,privacy_score\n2,4,0.8,0.25,75\n');await fs.mkdir(path.join(f.research,'downloads'));await fs.writeFile(path.join(f.research,'downloads','nih.zip.part'),'12345');
 const s=await(await f.request('/api/status')).json();assert.equal(s.results.length,1);assert.equal(s.results[0].privacy_score,'75');assert.equal(s.results[0].source,'stage_2_blur');assert.equal(s.downloads[0].bytes,5);assert.equal(s.downloads[0].partial,true);assert.equal(s.active,false);
}));
test('Host·Origin·본문·임의 경로를 거부하고 명령을 실행하지 않습니다',()=>fixture(async f=>{
 assert.equal((await f.request('/api/start',{method:'POST'})).status,403);
 assert.equal((await f.request('/api/start',{method:'POST',headers:{Origin:'https://outside.invalid'}})).status,403);
 const badHostStatus=await new Promise((resolve,reject)=>{const r=http.request(f.base+'/api/status',{headers:{Host:'outside.invalid'}},res=>{res.resume();resolve(res.statusCode);});r.on('error',reject);r.end();});
 assert.equal(badHostStatus,403);
 assert.equal((await f.request('/api/start',{method:'POST',headers:{Origin:f.base},body:'{}'})).status,400);
 assert.equal((await f.post('/api/start?command=calc')).status,404);
 assert.equal((await f.request('/server.mjs')).status,404);assert.equal(f.launches(),0);
}));
test('동시 시작과 이미 살아 있는 실험의 중복 실행을 막습니다',()=>fixture(async f=>{
 const responses=await Promise.all([f.post('/api/start'),f.post('/api/start')]);assert.deepEqual(responses.map(r=>r.status).sort(),[202,409]);assert.equal(f.launches(),1);
 assert.equal((await f.post('/api/start')).status,409);
}));
test('외부에서 시작한 실제 runner도 실행 ID 중단 요청을 받습니다',()=>fixture(async f=>{
 await fs.writeFile(path.join(f.control,'run.json'),JSON.stringify({pid:process.pid,run_id:'fixed-run-123',status:'running',phase:'download'}));
 assert.equal((await f.post('/api/start')).status,409);assert.equal(f.launches(),0);assert.equal((await f.post('/api/stop')).status,202);assert.equal(await fs.readFile(path.join(f.control,'stop.request'),'utf8'),'fixed-run-123');
}));
test('실행 기록 없는 중단은 거부하고 죽은 PID는 중단됨으로 표시합니다',()=>fixture(async f=>{
 assert.equal((await f.post('/api/stop')).status,409);await fs.writeFile(path.join(f.control,'run.json'),JSON.stringify({pid:2147483647,status:'running'}));const s=await(await f.request('/api/status')).json();assert.equal(s.run.status,'interrupted');assert.equal(s.active,false);
}));
test('로그는 상한을 지키고 UTF-8 첫 잘린 행을 제외합니다',()=>fixture(async f=>{
 const file=path.join(f.control,'experiment.log');const source='가'.repeat(40000)+'\n정상 마지막 행\n';assert.equal(Buffer.byteLength(source,'utf8'),120022);await fs.writeFile(file,source);const out=await tail(file,1024);assert.equal(out.truncated,true);assert.equal(out.text,'정상 마지막 행\n');assert.ok(!out.text.includes('\uFFFD'));assert.equal(out.bytes,120022);
}));
test('미리보기는 고정된 세 경로만 제공하고 경로 입력을 거부합니다',()=>fixture(async f=>{
 const previews=path.join(f.control,'previews');await fs.mkdir(previews);const image=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jQ1kAAAAASUVORK5CYII=','base64');
 assert.equal((await f.request('/api/preview/original.png')).status,404);
 for(const name of ['original','blur','mosaic']){await fs.writeFile(path.join(previews,name+'.png'),image);const res=await f.request('/api/preview/'+name+'.png');assert.equal(res.status,200);assert.equal(res.headers.get('content-type'),'image/png');assert.deepEqual(Buffer.from(await res.arrayBuffer()),image);}
 for(const request of ['/api/preview/other.png','/api/preview/original.png?path=../run.json','/api/preview/%2e%2e%2frun.json','/api/preview/../run.json'])assert.equal((await f.request(request)).status,404);
 const s=await(await f.request('/api/status')).json();assert.deepEqual(s.previews,{original:true,blur:true,mosaic:true});
}));
test('설치·다운로드와 평가를 구분하고 현재 실행 기록만 사용합니다',()=>{
 const base={active:true,run:{status:'running',phase:'CUDA 실행환경 설치 · 데이터 다운로드 병행'},results:[],log:{text:'[새 실행] 이전\n[단계 3] mosaic\n[새 실행] 현재\n[준비] 설치'}};
 assert.equal(describeExperiment(base).status,'실험 준비 중');assert.ok(describeExperiment(base).stages.every(s=>!s.active));
 base.run.phase='1→2→3단계 평가 실행';base.log.text+='\n[단계 2] blur / 강도 16';assert.equal(describeExperiment(base).status,'평가 실행 중');assert.equal(describeExperiment(base).stages[1].active,true);assert.equal(describeExperiment(base).stages[2].active,false);
});
test('실패한 평가와 평가 전 준비 실패를 구분합니다',()=>{
 const s={active:false,run:{status:'failed',phase:'실행 오류'},results:[],log:{text:'[새 실행] 현재\n[단계 2] blur'}};
 assert.equal(describeExperiment(s).stages[1].failed,true);assert.match(describeExperiment(s).stages[1].label,/평가 실패/);assert.ok(describeExperiment(s).stages.every(x=>!x.active));
 s.log.text='[새 실행] 현재\n[오류] CUDA 검사 실패';assert.ok(describeExperiment(s).stages.every(x=>x.label==='준비 실패로 대기'));
});
test('양쪽 테마의 모든 색 변수와 모션 축소·저장 설정이 있습니다',async()=>{
 const css=await fs.readFile(new URL('../화면/style.css',import.meta.url),'utf8');const js=await fs.readFile(new URL('../화면/app.js',import.meta.url),'utf8');
 const light=css.match(/:root\{([^}]+)\}/)[1];const dark=css.match(/:root\[data-theme=dark\]\{([^}]+)\}/)[1];const names=text=>[...text.matchAll(/(--[\w-]+):/g)].map(x=>x[1]).sort();assert.deepEqual(names(dark),names(light));
 for(const name of [...css.matchAll(/var\((--[\w-]+)\)/g)].map(x=>x[1]))assert.ok(names(light).includes(name),name);
 assert.match(css,/prefers-reduced-motion:reduce/);assert.match(css,/\.paused \.image-panel\{animation-play-state:paused\}/);assert.match(js,/localStorage.getItem\('cxr-theme'\)/);assert.match(js,/localStorage.setItem\('cxr-theme',theme\)/);assert.match(js,/timeZone:'Asia\/Seoul'/);
});
test('현재 학습 실행의 결과만 읽고 이전 고정 경로 결과를 섞지 않습니다',()=>fixture(async f=>{
 await fs.writeFile(path.join(f.control,'run.json'),JSON.stringify({run_id:'new-run',pid:process.pid,status:'running',phase:'검증 모델·검색 모델 처음부터 학습'}));
 for(const [folder,value] of [['results/stage_1_original','0.99'],['results/runs/old-run/stage_1_original','0.98'],['results/runs/new-run/stage_1_original','0.25']]){
  const target=path.join(f.research,folder);await fs.mkdir(target,{recursive:true});await fs.writeFile(path.join(target,'summary.csv'),`auc\n${value}\n`);
 }
 const s=await(await f.request('/api/status')).json();assert.equal(s.results.length,1);assert.equal(s.results[0].auc,'0.25');assert.equal(describeExperiment(s).status,'학습 시작 준비 중');assert.ok(describeExperiment(s).stages.every(x=>!x.active));
}));
test('실행 ID의 경로 탈출은 결과 읽기로 연결되지 않습니다',()=>fixture(async f=>{
 await fs.writeFile(path.join(f.control,'run.json'),JSON.stringify({run_id:'../../outside',status:'stopped'}));
 const s=await(await f.request('/api/status')).json();assert.equal(s.results.length,0);assert.equal(s.training,null);
}));
test('준비 전용 완료와 GPU 대기를 실제 학습·평가 완료로 표시하지 않습니다',()=>{
 const s={active:false,run:{status:'completed',mode:'데이터 준비 전용'},results:[],log:{text:''}};
 assert.equal(describeExperiment(s).status,'데이터 준비 완료');
 s.active=true;s.run={status:'running',phase:'검증 모델·검색 모델 처음부터 학습'};s.training={status:'waiting_for_gpu'};
 assert.equal(describeExperiment(s).status,'GPU 여유 메모리 대기');assert.ok(describeExperiment(s).stages.every(x=>!x.active));
});

test('모션 조작부는 상태별 크기를 고정하고 전역 호버 배경에서 분리합니다',async()=>{
 const css=await fs.readFile(new URL('../화면/style.css',import.meta.url),'utf8');
 assert.match(css,/\.motion\{[^}]*min-height:44px;min-width:44px;[^}]*flex-shrink:0;[^}]*background:transparent/);
 assert.match(css,/\.motion:hover,\.motion:active,\.motion:disabled\{background:transparent\}/);
 assert.match(css,/\.motion \.switch\{flex-shrink:0;/);
 assert.match(css,/\.motion:focus-visible\{outline:3px solid var\(--focus\);outline-offset:2px\}/);
 assert.match(css,/\.paused \.switch\{background:var\(--switch-off\)\}/);
 assert.match(css,/\.section-title\{flex-wrap:wrap\}/);
 for(const selector of ['.motion:hover','.motion:active','.motion:focus-visible','.motion:disabled']){
  const block=css.match(new RegExp(selector.replaceAll('.', '\\.')+'\\{([^}]+)\\}'));
  if(block)assert.doesNotMatch(block[1],/(?:^|;)(?:width|height|padding|right|left|transform):/);
 }
});

test('한글 화면 폴더에서만 고정 정적 파일을 제공합니다',()=>fixture(async f=>{
 const folder=path.join(f.dir,'화면');await fs.mkdir(folder);
 await fs.writeFile(path.join(folder,'index.html'),'<title>흉부 엑스레이 5차 검사</title>');
 const res=await f.request('/');assert.equal(res.status,200);assert.match(await res.text(),/흉부 엑스레이 5차 검사/);
 assert.match(res.headers.get('content-security-policy'),/default-src 'self'/);
 for(const name of ['/화면/index.html','/../server.mjs','/%2e%2e%2fserver.mjs'])assert.equal((await f.request(name)).status,404);
}));

test('6차 화면 상태와 실제 0회 갱신은 학습 성공으로 표시하지 않습니다',()=>fixture(async f=>{
 const status=await(await f.request('/api/status')).json();assert.equal(status.revision,6);
 const s={active:true,run:{status:'running',phase:'검증 모델·검색 모델 처음부터 학습'},results:[],log:{text:''},training:{phase:'verification',status:'training',global_step:0}};
 assert.equal(describeExperiment(s).status,'첫 가중치 갱신 대기');assert.ok(describeExperiment(s).stages.every(x=>!x.active&&!x.done));
 s.training.global_step=1;assert.equal(describeExperiment(s).status,'새 모델 학습 중');
 s.training={phase:'data_validation',status:'initializing',global_step:0};assert.equal(describeExperiment(s).status,'학습 데이터 검증 중');
}));

test('백분율은 십진 문자열에서 소수 첫째 자리로 반올림하고 단위를 구분합니다',async()=>{
 const {formatPercent}=await import('../화면/state.mjs');assert.equal(typeof formatPercent,'function');
 for(const [raw,expected] of [['0.9878','98.8%'],['0.987899999999999999','98.8%'],['0.999999999999999999999','100.0%'],['0.000099999','0.0%'],['0.0001','0.0%'],['1','100.0%'],['0','0.0%'],[0,'0.0%'],['9.878e-1','98.8%'],['9878e-4','98.8%'],['1e-10000','0.0%'],['-0.987899','-98.8%']])assert.equal(formatPercent(raw),expected,String(raw));
 assert.equal(formatPercent('98.789999',{ratio:false}),'98.8%');assert.equal(formatPercent('0.9878',{ratio:false}),'1.0%');
 assert.equal(formatPercent('9.878E+1',{ratio:false}),'98.8%');assert.equal(formatPercent('.9878'),'98.8%');assert.equal(formatPercent('-0.00001'),'0.0%');
 for(const raw of [null,undefined,'',' ',NaN,Infinity,-Infinity,'NaN','Infinity','1e9999',true,{},'0x10'])assert.equal(formatPercent(raw),'미측정',String(raw));
});

test('AUC·손실·학습률은 저장값 표현을 보존하고 원값 펼침은 안전한 텍스트입니다',async()=>{
 const {formatRaw}=await import('../화면/state.mjs');assert.equal(typeof formatRaw,'function');
 assert.equal(formatRaw('0.994012345678901234'),'0.994012345678901234');assert.equal(formatRaw('1.0000e-7'),'1.0000e-7');assert.equal(formatRaw(0),'0');
 for(const raw of [null,undefined,'',' ',NaN,Infinity,'nan',true])assert.equal(formatRaw(raw),'미측정');
 const js=await fs.readFile(new URL('../화면/app.js',import.meta.url),'utf8');const html=await fs.readFile(new URL('../화면/index.html',import.meta.url),'utf8');
 assert.doesNotMatch(js,/const metric\s*=.*toFixed/);assert.match(js,/formatPercent\(r\.privacy_score,\{ratio:false\}\)/);
 assert.match(js,/raw\.textContent=JSON\.stringify\(r,null,2\)/);assert.match(js,/createElement\('details'\)/);assert.doesNotMatch(js,/innerHTML/);
 assert.match(html,/id="training-raw"/);assert.match(html,/소수 첫째 자리로 반올림/);
 assert.match(js,/expanded\.has\(details\.dataset\.key\)/);
});

test('실제 CSV 소수 문자열·빈값·비유한 표기는 API 조회 후에도 원본 그대로입니다',()=>fixture(async f=>{
 const folder=path.join(f.research,'results/stage_2_blur');await fs.mkdir(folder,{recursive:true});
 const text='auc,cmc_rank_1,privacy_score,accuracy\n0.994012345678901234,0.987899999999999999,1.2100000000000001,\nNaN,0,Infinity,0\n';
 const file=path.join(folder,'summary.csv');await fs.writeFile(file,text);
 const status=await(await f.request('/api/status')).json();
 assert.equal(status.results[0].auc,'0.994012345678901234');assert.equal(status.results[0].cmc_rank_1,'0.987899999999999999');assert.equal(status.results[0].privacy_score,'1.2100000000000001');assert.equal(status.results[0].accuracy,'');
 assert.equal(status.results[1].auc,'NaN');assert.equal(status.results[1].cmc_rank_1,'0');assert.equal(status.results[1].privacy_score,'Infinity');assert.equal(await fs.readFile(file,'utf8'),text);
}));

test('21개 고정 조건은 모두 선택 가능하며 일치한 단일 행만 결과로 채택합니다',async()=>{
 const {EXPERIMENT_PLAN,conditionState}=await import('../화면/state.mjs');assert.equal(typeof conditionState,'function');
 assert.equal(EXPERIMENT_PLAN.reduce((n,p)=>n+p.strengths.length,0),21);
 assert.deepEqual(EXPERIMENT_PLAN[1].strengths,[0,2,4,8,16,32,64,128,256,512]);
 assert.deepEqual(EXPERIMENT_PLAN[2].strengths,[1,4,8,16,32,64,128,256,512,1024]);
 const s={active:true,run:{status:'running',phase:'검증 모델·검색 모델 처음부터 학습'},log:{text:'[새 실행] 현재'},results:[]};
 for(const plan of EXPERIMENT_PLAN)for(const strength of plan.strengths)assert.equal(conditionState(s,plan.stage,strength).status,'waiting');
 s.results=[{source:'stage_2_blur',stage:'2',mode:'blur',strength:'16',auc:'0.12345'}];
 assert.equal(conditionState(s,2,16).row.auc,'0.12345');assert.equal(conditionState(s,2,32).row,null);
 s.results.push({...s.results[0],auc:'0.999'});assert.equal(conditionState(s,2,16).status,'error');assert.equal(conditionState(s,2,16).row,null);
 assert.equal(describeExperiment(s).stages[1].done,false);
});

test('조건 활성은 명시 로그에만 연결되고 중복 행 10개로 단계를 완료하지 않습니다',async()=>{
 const {conditionState}=await import('../화면/state.mjs');
 const s={active:true,run:{status:'running',phase:'검증 모델·검색 모델 처음부터 학습'},log:{text:'[새 실행] 현재\n[단계 2] blur / 강도 16: 검증·검색 실행'},results:[]};
 assert.equal(conditionState(s,2,16).status,'waiting');s.run.phase='새 학습 가중치로 1→2→3단계 평가 실행';assert.equal(conditionState(s,2,16).status,'running');assert.equal(conditionState(s,2,32).status,'waiting');
 s.run.status='failed';s.active=false;assert.equal(conditionState(s,2,16).status,'error');assert.equal(conditionState(s,2,32).status,'waiting');
 s.results=Array.from({length:10},()=>({source:'stage_2_blur',stage:'2',mode:'blur',strength:'16'}));assert.equal(describeExperiment(s).stages[1].done,false);
});

test('반올림 100.0과 실제 원비율 1을 구분하며 십진 경계에서 정확히 반올림합니다',async()=>{
 const {formatPercent,percentNotice}=await import('../화면/state.mjs');assert.equal(typeof percentNotice,'function');
 for(const [value,expected] of [['0.9878048780487805','98.8%'],['0.987499999999999999','98.7%'],['0.9875','98.8%'],['9.995e-1','100.0%'],['0','0.0%'],['1','100.0%']])assert.equal(formatPercent(value),expected);
 assert.match(percentNotice('0.999999999999999999'),/반올림.*100% 미만/);assert.match(percentNotice('1.0000'),/정확히 1/);
 assert.match(percentNotice('99.999999',{ratio:false}),/100% 미만/);assert.match(percentNotice('100',{ratio:false}),/정확히 100/);
});

test('화면6차는 주입한 실험5차 상태·학습·중단만 연결하고 output 임의 경로를 읽지 않습니다',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'cxr-execution-'));const screen=path.join(dir,'6차'),execution=path.join(dir,'5차'),research=path.join(dir,'연구 자료'),control=path.join(execution,'work/control');
 await fs.mkdir(control,{recursive:true});const training=path.join(execution,'work/runs/current/training');await fs.mkdir(training,{recursive:true});
 await fs.writeFile(path.join(control,'run.json'),JSON.stringify({run_id:'current',pid:process.pid,status:'running',revision:5,output:'../../arbitrary'}));await fs.writeFile(path.join(training,'progress.json'),JSON.stringify({phase:'verification',loss:0.25}));
 const server=createDashboard({root:screen,executionRoot:execution,research,control,gpuQuery:(_a,_b,_c,cb)=>cb(null,'Test,0,0,1'),launch:()=>{throw Error('중복 시작 금지');}});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
 try{const s=await(await fetch(base+'/api/status')).json();assert.equal(s.revision,6);assert.equal(s.run.revision,5);assert.equal(s.training.loss,0.25);assert.equal(s.experiment.condition_count,21);assert.equal((await fetch(base+'/api/start',{method:'POST',headers:{Origin:base}})).status,409);assert.equal((await fetch(base+'/api/stop',{method:'POST',headers:{Origin:base}})).status,202);assert.equal(await fs.readFile(path.join(control,'stop.request'),'utf8'),'current');}
 finally{await new Promise(r=>server.close(r));await fs.rm(dir,{recursive:true,force:true});}
});

test('학습 phase는 평가와 구분하고 초기 best 가중치 파일만으로 완료를 표시하지 않습니다',async()=>{
 const {trainingPhaseStates}=await import('../화면/state.mjs');
 const s={active:true,run:{status:'running'},training:{phase:'verification',status:'training',global_step:0},training_phases:{verification:{initialization:{global_step:0},first_backward:null,selected_weights:true}}};
 let phases=trainingPhaseStates(s);assert.equal(phases[0].complete,false);assert.equal(phases[0].label,'첫 가중치 갱신 대기');assert.equal(phases[1].label,'학습 대기');
 s.training={phase:'data_validation',status:'validating'};assert.ok(trainingPhaseStates(s).every(p=>!p.active&&!p.complete));
 s.training={phase:'retrieval_phase1',status:'waiting_for_gpu'};phases=trainingPhaseStates(s);assert.equal(phases[0].complete,true);assert.equal(phases[1].label,'GPU 여유 메모리 대기');assert.equal(phases[2].complete,false);
});

test('선택은 허용된 hash만 사용하고 chip DOM을 갱신마다 재생성하지 않습니다',async()=>{
 const js=await fs.readFile(new URL('../화면/app.js',import.meta.url),'utf8');const html=await fs.readFile(new URL('../화면/index.html',import.meta.url),'utf8');
 assert.match(js,/history\.replaceState\(null,'',`#stage-/);assert.match(js,/strengths\.includes\(strength\)/);assert.match(js,/ArrowLeft/);assert.match(js,/aria-selected/);assert.doesNotMatch(js,/\$\('condition-chips'\)\.replaceChildren/);
 for(const id of ['condition-tabs','condition-chips','condition-detail','condition-raw','revision-label'])assert.match(html,new RegExp('id="'+id+'"'));
 assert.match(html,/확장 비교/);assert.match(html,/강도마다 다시 학습하지 않습니다/);
});

test('실패 원값 0은 보존하고 현재 실행의 마지막 정상 학습만 별도로 제공합니다',()=>fixture(async f=>{
 const id='failed-current',folder=path.join(f.dir,'work/runs',id,'training');await fs.mkdir(folder,{recursive:true});
 const original={phase:'failed',status:'failed',epoch:0,epochs:0,step:0,loss:null,error:'원래 CUDA 오류'};
 const observed={phase:'verification',status:'training',epoch:1,epochs:100,step:3326,global_step:3326,loss:0.5295830368995667,learning_rate:0.0001,updated_at:'2026-11-10T04:21:06.635882+00:00'};
 await fs.writeFile(path.join(f.control,'run.json'),JSON.stringify({run_id:id,status:'failed',error:'원래 실행 오류',output:'../other'}));
 await fs.writeFile(path.join(folder,'progress.json'),JSON.stringify(original));await fs.writeFile(path.join(folder,'events.jsonl'),JSON.stringify(observed)+'\n'+JSON.stringify(original)+'\n');
 const other=path.join(f.dir,'work/runs/other/training');await fs.mkdir(other,{recursive:true});await fs.writeFile(path.join(other,'events.jsonl'),JSON.stringify({...observed,global_step:9999})+'\n');
 const s=await(await f.request('/api/status')).json();assert.deepEqual(s.training,original);assert.deepEqual(s.last_observed_progress,observed);assert.equal(s.run.status,'failed');assert.equal(s.run.error,'원래 실행 오류');assert.equal(s.active,false);assert.equal(s.training_history.source,'training/events.jsonl');assert.equal(s.training_history.status,'ok');
 assert.equal(await fs.readFile(path.join(folder,'progress.json'),'utf8'),JSON.stringify(original));
 const {trainingDisplay,trainingPhaseStates}=await import('../화면/state.mjs');assert.equal(trainingDisplay(s).progress.global_step,3326);assert.equal(trainingDisplay(s).lastObserved,true);
 const phases=trainingPhaseStates(s);assert.ok(phases.every(p=>!p.active));assert.equal(phases[0].lastObserved,true);assert.equal(phases[0].progress.step,3326);assert.match(phases[0].label,/중단 또는 오류/);
}));

test('학습 JSONL은 LF 경계를 지키고 중간 손상·다른 실행·비유한 학습값을 거부합니다',async()=>{
 const {readTrainingHistory}=await import('../server.mjs');const folder=await fs.mkdtemp(path.join(os.tmpdir(),'cxr-history-'));const file=path.join(folder,'events.jsonl');
 const row={phase:'verification',status:'training',epoch:0,epochs:100,step:0,global_step:0,loss:0,learning_rate:0,updated_at:'2026-11-10T04:21:06Z',note:'앞\u0085뒤'};
 try{
  assert.equal((await readTrainingHistory(file,'current')).status,'missing');
  for(const [middle,status] of [['{broken','malformed'],['','malformed'],[JSON.stringify({...row,run_id:'other'}),'different_run'],[JSON.stringify({...row,loss:'NaN'}),'invalid_training_record'],[JSON.stringify({...row,global_step:-1}),'invalid_training_record']]){
   await fs.writeFile(file,JSON.stringify(row)+'\n'+middle+'\n');const result=await readTrainingHistory(file,'current');assert.equal(result.status,status);assert.equal(result.last_observed_progress,null);
  }
  await fs.writeFile(file,JSON.stringify(row)+'\n');const zero=await readTrainingHistory(file,'current');assert.deepEqual(zero.last_observed_progress,row);assert.equal(zero.status,'ok');
  await fs.writeFile(file,Buffer.concat([Buffer.from(JSON.stringify(row)+'\n'),Buffer.from([0xff,10])]));const bad=await readTrainingHistory(file,'current');assert.equal(bad.status,'malformed');assert.equal(bad.last_observed_progress,null);
 }finally{await fs.rm(folder,{recursive:true,force:true});}
});

test('JSONL 꼬리의 미완성 행과 앞부분 생략을 명시하고 읽기 상한을 지킵니다',async()=>{
 const {readTrainingHistory}=await import('../server.mjs');const folder=await fs.mkdtemp(path.join(os.tmpdir(),'cxr-tail-'));const file=path.join(folder,'events.jsonl');
 const row={phase:'retrieval_phase1',status:'training',epoch:2,epochs:30,step:7,global_step:9,loss:0.2,learning_rate:0.01,updated_at:'2026-11-10T04:21:06Z'};
 try{
  await fs.writeFile(file,'가'.repeat(150000)+'\n'+JSON.stringify(row)+'\n{"phase":');
  const result=await readTrainingHistory(file,'current');assert.equal(result.truncated,true);assert.equal(result.incomplete_tail,true);assert.equal(result.status,'incomplete_tail');assert.deepEqual(result.last_observed_progress,row);assert.ok(result.bytes>256*1024);
  await fs.writeFile(file,'x'.repeat(300000));const long=await readTrainingHistory(file,'current');assert.equal(long.truncated,true);assert.equal(long.last_observed_progress,null);
  await fs.writeFile(file,JSON.stringify(row));const partial=await readTrainingHistory(file,'current');assert.equal(partial.status,'incomplete_tail');assert.equal(partial.last_observed_progress,null);
 }finally{await fs.rm(folder,{recursive:true,force:true});}
});

test('진행 중에는 현재 값이 우선이고 마지막 기록 누락은 0회 갱신으로 해석하지 않습니다',async()=>{
 const {trainingDisplay,trainingPhaseStates}=await import('../화면/state.mjs');
 const s={active:true,run:{status:'running'},training:{phase:'verification',global_step:0,loss:0},last_observed_progress:{phase:'retrieval_phase2',global_step:900}};
 assert.equal(trainingDisplay(s).progress.global_step,0);assert.equal(trainingDisplay(s).lastObserved,false);assert.equal(trainingPhaseStates(s)[2].active,false);
 s.run.status='failed';s.active=false;s.last_observed_progress=null;assert.equal(trainingDisplay(s).progress,null);assert.equal(trainingDisplay(s).terminal,true);assert.ok(trainingPhaseStates(s).every(p=>!p.active));
 const js=await fs.readFile(new URL('../화면/app.js',import.meta.url),'utf8');assert.match(js,/마지막 관측/);assert.match(js,/현재 학습 아님/);assert.match(js,/current_progress:s.training,last_observed_progress/);assert.match(js,/0회 학습을 뜻하지 않습니다/);
});
