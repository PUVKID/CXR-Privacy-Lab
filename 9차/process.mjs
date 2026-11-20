import path from 'node:path';
import {execFile} from 'node:child_process';
// 경로 비교를 위한 좁은 Windows 인자 분석입니다. 따옴표가 닫히지 않으면 거부합니다.
export function commandTokens(command){
 if(typeof command!=='string')return null;
 const tokens=[];let token='',quoted=false,started=false;
 for(const c of command){if(c==='"'){quoted=!quoted;started=true;}else if(/\s/.test(c)&&!quoted){if(started){tokens.push(token);token='';started=false;}}else{token+=c;started=true;}}
 if(quoted)return null;if(started)tokens.push(token);return tokens;
}
const normalized=value=>typeof value==='string'?path.win32.normalize(value.replaceAll('/','\\')).toLowerCase():'';
function option(args,name){const indexes=args.map((value,index)=>value.toLowerCase()===name.toLowerCase()?index:-1).filter(index=>index>=0);return indexes.length===1?args[indexes[0]+1]:null;}
export function matchProcesses(rows,executionRoot,runId,pointerCreatedAt,now=Date.now()){
 const records=Array.isArray(rows)?rows:rows?[rows]:[];
 const byId=new Map(records.map(row=>[row.pid,row]));
 const scripts=new Map(['train_from_scratch.py','cxr7_evaluate.py','cxr7_privacy.py'].map(name=>[normalized(path.join(executionRoot,'src',name)),name]));
 const output=normalized(path.join(executionRoot,'runs',runId)),research=normalized(path.join(executionRoot,'../연구 자료'));
 const validDate=row=>Number.isFinite(Date.parse(row.created_at))&&Date.parse(row.created_at)<=now+5000;
 const candidates=records.filter(row=>{
  const args=commandTokens(row.command_line);if(!args||!Number.isInteger(row.pid)||!Number.isInteger(row.ppid)||!validDate(row))return false;
  const script=args.filter(value=>scripts.has(normalized(value)));if(script.length!==1)return false;
  const start=Date.parse(pointerCreatedAt);if(Number.isFinite(start)&&Date.parse(row.created_at)<start)return false;
  return normalized(option(args,'--output'))===output&&normalized(option(args,'--project'))===research;
 });
 const ids=new Set(candidates.map(row=>row.pid));
 const leaves=candidates.filter(row=>!candidates.some(child=>child.ppid===row.pid));
 const proofs=[];
 for(const leaf of leaves){
  const chain=[leaf];let node=leaf;const visited=new Set([leaf.pid]);
  while(ids.has(node.ppid)&&!visited.has(node.ppid)){const parent=byId.get(node.ppid);if(Date.parse(parent.created_at)>Date.parse(node.created_at)){node=null;break;}visited.add(parent.pid);chain.push(parent);node=parent;}
  if(!node)continue;
  const manager=byId.get(node.ppid),args=commandTokens(manager?.command_line);
  if(!manager||!args||!validDate(manager)||Date.parse(manager.created_at)>Date.parse(node.created_at))continue;
  if(normalized(option(args,'-File'))!==normalized(path.join(executionRoot,'All-In-One.ps1'))||option(args,'-RunID')!==runId||!['resume','evaluate','privacy'].includes(option(args,'-Action')))continue;
  const leafArgs=commandTokens(leaf.command_line),script=leafArgs.find(value=>scripts.has(normalized(value)));
  proofs.push({pid:leaf.pid,created_at:leaf.created_at,manager_pid:manager.pid,manager_created_at:manager.created_at,redirector_pids:chain.slice(1).map(row=>row.pid),script:scripts.get(normalized(script))});
 }
 return {status:proofs.length===1&&leaves.length===1?'confirmed':proofs.length>1||leaves.length>1?'ambiguous':candidates.length?'unconfirmed':'not_found',verified:proofs.length===1&&leaves.length===1,proofs,candidate_count:candidates.length,leaf_count:leaves.length};
}
const QUERY="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); @(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'powershell.exe' OR Name = 'pwsh.exe'\" -ErrorAction Stop | ForEach-Object { @{pid=[int]$_.ProcessId;ppid=[int]$_.ParentProcessId;created_at=$_.CreationDate.ToUniversalTime().ToString('o');command_line=$_.CommandLine} }) | ConvertTo-Json -Compress";
export function createProcessObserver(query){
 let at=0,cached=null,pending=null;
 const read=query||(()=>new Promise((resolve,reject)=>execFile(path.join(process.env.SystemRoot||'C:/Windows','System32/WindowsPowerShell/v1.0/powershell.exe'),['-NoLogo','-NoProfile','-NonInteractive','-Command',QUERY],{windowsHide:true,timeout:4000,maxBuffer:1024*1024,encoding:'utf8'},(error,text)=>{if(error)reject(error);else{try{resolve(JSON.parse(text.replace(/^\uFEFF/,'')));}catch(error){reject(error);}}})));
 return async function observe(root,id,created){
  if(!cached||Date.now()-at>10000){if(!pending)pending=read().then(rows=>{cached={rows};at=Date.now();}).catch(()=>{cached={error:true};at=Date.now();}).finally(()=>{pending=null;});await pending;}
  return {...(cached.error?{status:'unavailable',verified:false,proofs:[]}:matchProcesses(cached.rows,root,id,created)),observed_at:new Date(at).toISOString(),cache_seconds:10};
 };
}
