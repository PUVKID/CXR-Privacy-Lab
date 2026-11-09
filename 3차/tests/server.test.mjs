import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import http from 'node:http';
import { createDashboard, parseCSV, tail } from '../server.mjs';
import { describeExperiment } from '../app/state.mjs';

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
 assert.equal(describeExperiment(s).stages[1].failed,true);assert.equal(describeExperiment(s).stages[1].label,'평가 실패');assert.ok(describeExperiment(s).stages.every(x=>!x.active));
 s.log.text='[새 실행] 현재\n[오류] CUDA 검사 실패';assert.ok(describeExperiment(s).stages.every(x=>x.label==='준비 실패로 대기'));
});
test('양쪽 테마의 모든 색 변수와 모션 축소·저장 설정이 있습니다',async()=>{
 const css=await fs.readFile(new URL('../app/style.css',import.meta.url),'utf8');const js=await fs.readFile(new URL('../app/app.js',import.meta.url),'utf8');
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
