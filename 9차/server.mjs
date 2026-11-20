import http from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { readSevenStatus,validRunId,previewName } from './adapter.mjs';
import {createProcessObserver} from './process.mjs';
import { DATASETS,CONDITIONS } from './화면/state.mjs';
const ROOT=path.dirname(fileURLToPath(import.meta.url));
export function createDashboard(options={}){
 const root=options.root||ROOT,executionRoot=options.executionRoot||path.resolve(ROOT,'../7차');
 const observe=createProcessObserver(options.processQuery);
 let gpu={name:'확인 중',utilization:null},gpuAt=0,cached=null,cachedAt=0,pending=null;
 async function status(){
  if(Date.now()-gpuAt>15000){gpuAt=Date.now();(options.gpuQuery||execFile)('nvidia-smi',['--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],{windowsHide:true,timeout:2000,maxBuffer:4096},(error,out)=>{
   if(error)gpu={name:'조회 불가',utilization:null};else{const [name,utilization,used,total]=out.trim().split('\n')[0].split(',').map(x=>x.trim());gpu={name,utilization,used,total};}
  });}
  if(cached&&Date.now()-cachedAt<500)return {...cached,gpu,process:cached.run_id?await observe(executionRoot,cached.run_id,cached.pointer_created_at):{status:'unavailable',verified:false,proofs:[]}};
  if(!pending)pending=readSevenStatus(executionRoot).then(value=>{cached=value;cachedAt=Date.now();return value;}).finally(()=>{pending=null;});
  const s=await pending;return {...s,gpu,process:s.run_id?await observe(executionRoot,s.run_id,s.pointer_created_at):{status:'unavailable',verified:false,proofs:[]}};
 }
 const server=http.createServer(async(req,res)=>{
  const send=(code,value)=>{res.writeHead(code,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
  try{
   const host=`127.0.0.1:${server.address().port}`;
   if(req.headers.host!==host)return send(403,{error:'로컬 Host만 허용합니다.'});
   // 9차 화면에는 실험 생성·종료·파일 쓰기 요청이 없습니다.
   if(req.method!=='GET')return send(405,{error:'읽기 전용 대시보드입니다. 실험 제어 요청은 제공하지 않습니다.'});
   if(req.url==='/api/status')return send(200,await status());
   const preview=/^\/api\/preview\/(NIH|COVID19)\/([a-z]+_\d{3})\.png$/.exec(req.url);
   if(preview&&DATASETS.includes(preview[1])&&CONDITIONS.some(item=>item.id===preview[2])){
    const s=await status();if(!validRunId(s.run_id)||!s.previews?.[preview[1]]?.[preview[2]])return send(404,{error:'현재 실행의 대표사진이 아직 없습니다.'});
    const base=await fs.realpath(path.join(executionRoot,'runs',s.run_id));
    const file=path.resolve(base,'대표사진',preview[1],previewName(preview[2]));
    if(await fs.realpath(file)!==file)return send(404,{error:'대표사진 경계 확인 실패'});
    const info=await fs.stat(file);if(!info.isFile()||info.size>12*1024*1024)return send(404,{error:'대표사진 크기 제한'});
    const data=await fs.readFile(file);res.writeHead(200,{'Content-Type':'image/png','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});return res.end(data);
   }
   const files={'/':['index.html','text/html'],'/style.css':['style.css','text/css'],'/app.js':['app.js','text/javascript'],'/state.mjs':['state.mjs','text/javascript'],'/assets/NanumGothic-Bold.ttf':['assets/NanumGothic-Bold.ttf','font/ttf'],'/assets/OFL.txt':['assets/OFL.txt','text/plain']};
   const entry=files[req.url];if(!entry)return send(404,{error:'없는 경로입니다.'});
   const data=await fs.readFile(path.join(root,'화면',entry[0]));res.writeHead(200,{'Content-Type':entry[1]+'; charset=utf-8','Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'});res.end(data);
  }catch(error){send(500,{error:'로컬 기록을 읽을 수 없습니다. 원본은 변경하지 않았습니다.'});}
 });
 return server;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const server=createDashboard();server.on('error',error=>{console.error(`[대시보드 시작 실패] ${error.code}: ${error.message}`);process.exitCode=1;});
 server.listen(8787,'127.0.0.1',()=>console.log('화면 9차 · 실험 7차 · 읽기 전용: http://127.0.0.1:8787'));
}
