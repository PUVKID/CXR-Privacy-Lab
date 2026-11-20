"""7차의 실제 정상 종료를 확인한 뒤20개의 새 학습을 한 번에 하나씩 실행합니다."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from cxr10_contract import (PROTOCOL_NAME, atomic_json, canonical_digest, condition,
    condition_fingerprint, condition_plan, evaluation_argv, file_sha256,
    initial_queue_state, plain_path, read_json, run_relative, training_argv,
    utc_now, valid_run_id, validate_training_identity)


def baseline_exit_gate(exit_code):
    if exit_code is None:
        return 'waiting_baseline'
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError('원본 관리자의 실제 정상 exit0가 없어 새 학습을 시작하지 않습니다.')
    return 'ready'


def same_process(expected, actual):
    return all(expected.get(key) is not None and expected[key] == actual.get(key)
               for key in ('pid', 'creation_filetime', 'command_line'))


def require_child_success(exit_code, condition_id):
    if type(exit_code) is not int or exit_code != 0:
        raise RuntimeError(f'{condition_id} 자식 실행 실패: exit={exit_code}; 다음 조건은 시작하지 않습니다.')


class QueueLock:
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        plain_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        try:
            if self.path.stat().st_size == 0:
                self.file.write(b'0')
                self.file.flush()
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            self.file = None
            raise
        return self

    def __exit__(self, *ignored):
        if self.file is not None:
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None


def kernel32():
    if os.name != 'nt':
        raise RuntimeError('실제 큐 프로세스 검증은 Windows에서만 지원합니다.')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    api.GetProcessTimes.restype = wintypes.BOOL
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.GetExitCodeProcess.restype = wintypes.BOOL
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


def process_creation(api, handle):
    stamps = [wintypes.FILETIME() for _ in range(4)]
    if not api.GetProcessTimes(handle, *(ctypes.byref(value) for value in stamps)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (stamps[0].dwHighDateTime << 32) | stamps[0].dwLowDateTime


def filetime_utc(value):
    return (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(value) // 10)).isoformat()


def cim_process(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('프로세스 ID가 잘못됐습니다.')
    # 고정 숫자PID만 쿼리에 넣으며 스크립트 파일/실행정책/시스템 설정은 변경하지 않습니다.
    script = ("$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "';"
        "if($null -eq $p){throw '대상 프로세스 없음'};"
        "@{pid=[int]$p.ProcessId;parent_pid=[int]$p.ParentProcessId;command_line=[string]$p.CommandLine;"
        "created_utc=$p.CreationDate.ToUniversalTime().ToString('o')}|ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
        capture_output=True, timeout=15, check=False)
    if result.returncode or len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise RuntimeError('프로세스 신원 조회 실패: ' + result.stderr[:2000].decode('utf-8', 'replace'))
    value = json.loads(result.stdout.decode('utf-8-sig', 'strict'))
    if value.get('pid') != pid or not value.get('command_line') or not value.get('created_utc'):
        raise ValueError('현재 프로세스의 생성시각·명령을 확인할 수 없습니다.')
    return value


class ProcessHandle:
    def __init__(self, pid, expected=None):
        self.api, self.handle = kernel32(), None
        handle = self.api.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        try:
            created = process_creation(self.api, handle)
            current = cim_process(pid)
            current['creation_filetime'] = str(created)
            kernel_date = datetime.fromisoformat(filetime_utc(created))
            cim_date = datetime.fromisoformat(current['created_utc'].replace('Z', '+00:00'))
            if abs((kernel_date - cim_date).total_seconds()) >= .001:
                raise ValueError('CIM과 보유 핸들의 프로세스 생성시각이 다릅니다.')
            self.identity = current
            if expected is not None and not same_process(expected, current):
                raise ValueError('관리자 PID가 재사용됐거나 명령/생성시각이 달라졌습니다.')
        except BaseException:
            self.close()
            raise

    def poll(self):
        status = self.api.WaitForSingleObject(self.handle, 0)
        if status == 258:
            return None
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.api.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(code.value)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.close()


@contextmanager
def prevent_sleep():
    api = kernel32()
    api.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    api.SetThreadExecutionState.restype = wintypes.DWORD
    if not api.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        api.SetThreadExecutionState(0x80000000)


def windows_argv(command):
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not pointer:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [pointer[i] for i in range(count.value)]
    finally:
        api = kernel32()
        api.LocalFree.argtypes = [wintypes.HLOCAL]
        api.LocalFree.restype = wintypes.HLOCAL
        api.LocalFree(pointer)


def validate_baseline_invocation(identity, root, run_id):
    argv = windows_argv(identity['command_line'])
    lower = [token.casefold() for token in argv]
    def option(name):
        if lower.count(name.casefold()) != 1:
            raise ValueError('원본 관리자 인자 누락/중복: ' + name)
        index = lower.index(name.casefold())
        if index + 1 >= len(argv):
            raise ValueError('원본 관리자 인자 값 없음')
        return argv[index + 1]
    if (Path(option('-file')).resolve() != (root.parent / '7차/All-In-One.ps1').resolve()
            or option('-action').casefold() not in ('start', 'resume')
            or option('-runid') != run_id):
        raise ValueError('현재7차 관리자의 정확한 실행 소유권이 다릅니다.')


def source_fingerprint(root):
    return canonical_digest({p.relative_to(root).as_posix(): file_sha256(p)
        for area in ('src', 'official_source') for p in sorted((root / area).rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts})


def baseline_artifact_gate(root, run_id):
    """학습의 complete 표시만으로 gate를 열지 않고 원본84개 평가도 검사합니다."""
    from cxr7_privacy_contract import condition_plan as baseline_plan, report_is_complete, validate_baseline
    from cxr7_reports import evaluator_fingerprint
    from cxr7_contract import EXPECTED_SHA256
    seven = root.parent / '7차'
    output = seven / 'runs' / valid_run_id(run_id)
    plain_path(output)
    artifacts = read_json(output / 'artifacts.json')
    evaluation = read_json(output / 'evaluation_results.json')
    privacy = read_json(output / 'privacy_results.json')
    hashes = {task: file_sha256(output / phase / 'selected_weights.pth')
        for task, phase in (('verification', 'verification'), ('retrieval', 'retrieval_phase2'))}
    code_hashes = evaluator_fingerprint(seven / 'src')
    validate_baseline(artifacts, evaluation, hashes, evaluator_code_sha256=code_hashes)
    for dataset in ('NIH', 'COVID19'):
        repeats = evaluation[dataset]['verification'].get('confidence_interval', {}).get('bootstrap_repeats')
        if type(repeats) is not int or repeats < 10000:
            raise ValueError('원본 검증 평가의 지정 bootstrap10000 완료 근거가 없습니다.')
    if privacy.get('completed') is not True or not report_is_complete(privacy):
        raise ValueError('7차 원본 고정모델 평가84과제가 아직 모두 완료되지 않았습니다.')
    payload = privacy.get('binding_payload', {})
    # 기존7차 open_report의 JSON 직렬화 방법으로 원래 지문을 검증합니다.
    import hashlib
    original_binding = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode('utf-8')).hexdigest()
    if privacy.get('binding') != original_binding or payload.get('checkpoints_sha256') != hashes:
        raise ValueError('원본84평가의 모델/프로토콜 지문이 다릅니다.')
    if payload.get('baseline_results_sha256') != file_sha256(output / 'evaluation_results.json'):
        raise ValueError('원본평가와84평가 결과의 결합이 다릅니다.')
    if payload.get('plan') != baseline_plan():
        raise ValueError('기존21조건 계획이 다릅니다.')
    if payload.get('official_manifests') != EXPECTED_SHA256:
        raise ValueError('원본84평가의 공식 분할 지문이 다릅니다.')
    expected_sources = {'cxr7_privacy.py', 'cxr7_privacy_contract.py', 'cxr7_examples.py',
        'cxr_reid/evaluation.py', 'cxr_reid/inference.py', 'cxr_reid/image_ops.py',
        'cxr_reid/models.py', 'cxr_reid/data.py'}
    if set(payload.get('sources', {})) != expected_sources:
        raise ValueError('원본84평가의 필수 소스 결합이 누락됐습니다.')
    for name, expected in payload.get('sources', {}).items():
        candidate = seven / 'src' / name
        if not candidate.resolve().is_relative_to((seven / 'src').resolve()) or file_sha256(candidate) != expected:
            raise ValueError('원본 평가 소스가 변경됐습니다.')
    if not payload.get('sources'):
        raise ValueError('원본 평가 소스 결합이 없습니다.')
    return {'run_id': run_id, 'artifacts_sha256': file_sha256(output / 'artifacts.json'),
        'evaluation_sha256': file_sha256(output / 'evaluation_results.json'),
        'privacy_sha256': file_sha256(output / 'privacy_results.json'),
        'checkpoint_sha256': hashes, 'completed_task_conditions': 84}


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], capture_output=True, timeout=15, check=False)
    if result.returncode or len(result.stdout) > 65536:
        raise RuntimeError('GPU 현재 실행 목록을 확인하지 못해 시작하지 않습니다.')
    values = result.stdout.decode('utf-8', 'strict').strip().splitlines()
    if any(not line.strip().isdigit() for line in values):
        raise ValueError('GPU 실행 목록 응답을 안전하게 해석할 수 없습니다.')
    return [int(line.strip()) for line in values]


def completed_condition_gate(root, row):
    output = root / run_relative(row['run_id'])
    for stage in ('training', 'evaluation'):
        receipt = read_json(output / (stage + '_exit.json'))
        if receipt.get('condition_id') != row['id'] or receipt.get('run_id') != row['run_id']:
            raise ValueError('다른 조건·실행의 자식 종료 영수증입니다.')
        require_child_success(receipt.get('exit_code'), row['id'])
    config = read_json(output / 'training_config.json')
    artifacts = read_json(output / 'artifacts.json')
    results = read_json(output / 'evaluation_results.json')
    for value in (config, artifacts, results):
        validate_training_identity(value, row['id'], row['run_id'])
    from cxr7_privacy_contract import validate_baseline
    hashes = {task: file_sha256(output / phase / 'selected_weights.pth')
        for task, phase in (('verification', 'verification'), ('retrieval', 'retrieval_phase2'))}
    validate_baseline(artifacts, results, hashes)
    if not config.get('binding') or artifacts.get('binding') != config['binding'] or results.get('training_binding') != config['binding']:
        raise ValueError('조건 학습·가중치·평가의 전체 프로토콜 지문이 다릅니다.')
    return {'artifacts_sha256': file_sha256(output / 'artifacts.json'),
        'evaluation_sha256': file_sha256(output / 'evaluation_results.json'),
        'protocol_fingerprint': config['binding']}


def write_state(root, state):
    state['updated_at'] = utc_now()
    atomic_json(root / 'work/queue/state.json', state)


def validate_state(state, plan):
    if state.get('schema_version') != 1 or state.get('queue_id') != plan['queue_id']:
        raise ValueError('현재 큐 상태의 실행 결합이 다릅니다.')
    rows = state.get('conditions', [])
    if len(rows) != 20 or [row.get('id') for row in rows] != [row['id'] for row in condition_plan()]:
        raise ValueError('큐의20조건 목록이 변조됐습니다.')
    runs = set()
    for row in rows:
        expected = condition(row['id'])
        if any(type(row.get(key)) is not type(value) or row.get(key) != value for key, value in expected.items()) or row.get('condition_fingerprint') != condition_fingerprint(row['id']):
            raise ValueError('큐 조건 정의가 다릅니다.')
        if row.get('status') not in ('pending', 'training', 'evaluating', 'completed', 'failed', 'blocked'):
            raise ValueError('큐 조건 상태가 잘못됐습니다.')
        if row.get('run_id') is not None:
            if row['run_id'] in runs or row.get('training_subdir') != run_relative(row['run_id']):
                raise ValueError('중복/잘못된 조건 실행 경로입니다.')
            runs.add(row['run_id'])
        elif row.get('training_subdir') is not None or row['status'] != 'pending':
            raise ValueError('실행 ID 없이 조건이 시작된 것으로 표시됐습니다.')


def validate_plan(root, plan):
    if plan.get('schema_version') != 1 or plan.get('protocol') != PROTOCOL_NAME:
        raise ValueError('알 수 없는 큐 계획입니다.')
    valid_run_id(plan.get('queue_id'))
    baseline = plan.get('baseline', {})
    valid_run_id(baseline.get('run_id'))
    identity = baseline.get('process', {})
    if (type(identity.get('pid')) is not int or identity['pid'] <= 0
            or not isinstance(identity.get('creation_filetime'), str)
            or not identity['creation_filetime'].isdigit() or int(identity['creation_filetime']) <= 0
            or not isinstance(identity.get('command_line'), str) or not identity['command_line']):
        raise ValueError('원본 프로세스의 정확한 신원이 없습니다.')
    for key, expected in (('python', root.parent / '실행 환경/Scripts/python.exe'),
                          ('project', root.parent / '연구 자료')):
        actual = Path(plan.get(key, ''))
        plain_path(actual)
        if not actual.is_absolute() or actual.resolve() != expected.resolve():
            raise ValueError('큐의 실행기·원본 경로를 임의로 바꿀 수 없습니다.')
    if not re.fullmatch('[0-9a-f]{64}', plan.get('source_fingerprint', '')):
        raise ValueError('큐의 고정 소스 지문이 없습니다.')


def run_child(root, state, row, argv, stage):
    output = root / run_relative(row['run_id'])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / (stage + '_launch.json'), {'argv': argv, 'cwd': str(root),
        'condition_id': row['id'], 'run_id': row['run_id'], 'launched_at': utc_now()})
    with (output / (stage + '.stdout.log')).open('ab', buffering=0) as log:
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, shell=False, creationflags=flags)
        try:
            with ProcessHandle(child.pid) as handle:
                row['process'] = {**handle.identity, 'entry': Path(argv[5]).name,
                    'condition_id': row['id'], 'condition_fingerprint': row['condition_fingerprint'],
                    'run_id': row['run_id']}
                write_state(root, state)
                while child.poll() is None:
                    # 중단 표시는 다음 조건의 시작을 막으며 현재 소유 자식은 종료까지 유지합니다.
                    time.sleep(5)
                exit_code = child.wait()
                if handle.poll() != exit_code:
                    raise ValueError('보유 핸들과 자식 프로세스의 실제 종료 코드가 다릅니다.')
        except BaseException:
            # 신원을 확인할 수 없으면 프로세스를 종료하지 않고 PID와 로그를 보존하며 새 작업을 막습니다.
            row['unconfirmed_child_pid'] = child.pid
            write_state(root, state)
            raise
    row['last_child_exit'] = exit_code
    row['process'] = None
    atomic_json(output / (stage + '_exit.json'), {'exit_code': exit_code,
        'condition_id': row['id'], 'run_id': row['run_id'], 'finished_at': utc_now()})
    require_child_success(exit_code, row['id'])


def wait_baseline(root, plan, state, interval):
    receipt_path = root / 'work/queue/baseline_exit.private.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get('queue_id') != plan['queue_id'] or not same_process(plan['baseline']['process'], receipt.get('process', {})):
            raise ValueError('기존 종료 영수증의 관리자 신원이 다릅니다.')
        if receipt.get('source') != 'held_windows_process_handle' or type(receipt.get('exit_code')) is not int:
            raise ValueError('보유 핸들에서 얻은 실제 종료 코드 영수증이 없습니다.')
        baseline_exit_gate(receipt['exit_code'])
        return
    expected = plan['baseline']['process']
    with ProcessHandle(expected['pid'], expected) as handle:
        while True:
            code = handle.poll()
            if code is not None:
                atomic_json(receipt_path, {'queue_id': plan['queue_id'], 'process': expected,
                    'exit_code': code, 'observed_at': utc_now(), 'source': 'held_windows_process_handle'})
                baseline_exit_gate(code)
                return
            state['baseline']['status'] = 'running'
            write_state(root, state)
            if (root / 'work/queue/STOP_REQUESTED').exists():
                raise KeyboardInterrupt('대기 큐 중단 요청; 원본 프로세스는 그대로 유지합니다.')
            time.sleep(interval)


def prepare(root, args):
    valid_run_id(args.baseline_run_id)
    with QueueLock(root / 'work/queue/queue.lock'):
        plan_path = root / 'work/queue/plan.private.json'
        if plan_path.exists():
            raise FileExistsError('큐 계획이 이미 있습니다. 기존 계획을 덮어쓰지 않습니다.')
        with ProcessHandle(args.baseline_pid) as handle:
            validate_baseline_invocation(handle.identity, root, args.baseline_run_id)
            actual = datetime.fromisoformat(handle.identity['created_utc'].replace('Z', '+00:00'))
            expected = datetime.fromisoformat(args.baseline_created_utc.replace('Z', '+00:00'))
            if expected.tzinfo is None or abs((actual - expected).total_seconds()) >= .001:
                raise ValueError('부모가 지정한 기존 관리자 생성시각과 다릅니다.')
            baseline = {'run_id': args.baseline_run_id, 'status': 'running', 'process': handle.identity}
        plan = {'schema_version': 1, 'queue_id': str(uuid.uuid4()), 'baseline': baseline,
            'source_fingerprint': source_fingerprint(root), 'created_at': utc_now(),
            'python': str(root.parent / '실행 환경/Scripts/python.exe'),
            'project': str(root.parent / '연구 자료'), 'protocol': PROTOCOL_NAME}
        atomic_json(plan_path, plan)
        write_state(root, initial_queue_state(plan['queue_id'], baseline))
        return plan


def execute(root, args):
    with QueueLock(root / 'work/queue/queue.lock'), prevent_sleep():
        plan = read_json(root / 'work/queue/plan.private.json')
        state = read_json(root / 'work/queue/state.json')
        validate_plan(root, plan)
        validate_state(state, plan)
        if source_fingerprint(root) != plan['source_fingerprint']:
            raise ValueError('대기열 준비 이후 학습/공식목록 소스가 바뀌어 실행하지 않습니다.')
        with ProcessHandle(os.getpid()) as manager:
            state['manager'] = manager.identity
        current = None
        try:
            state['status'] = 'waiting_baseline'
            write_state(root, state)
            wait_baseline(root, plan, state, args.poll_seconds)
            proof = baseline_artifact_gate(root, plan['baseline']['run_id'])
            state['baseline'].update(status='completed', completion=proof)
            state['status'] = 'ready'
            write_state(root, state)
            for current in state['conditions']:
                if current['status'] == 'completed':
                    completed_condition_gate(root, current)
                    continue
                if (root / 'work/queue/STOP_REQUESTED').exists():
                    raise KeyboardInterrupt('현재 조건 뒤의 새 학습 시작을 중단합니다.')
                if current['run_id'] is not None and args.resume_condition != current['id']:
                    raise ValueError('기존 실패/중단 조건은 정확한 --resume-condition을 명시해야 합니다.')
                resume = current['run_id'] is not None
                if not resume:
                    current['run_id'] = str(uuid.uuid4())
                    current['training_subdir'] = run_relative(current['run_id'])
                    (root / current['training_subdir']).mkdir(parents=True, exist_ok=False)
                else:
                    validate_training_identity(read_json(root / current['training_subdir'] / 'training_config.json'),
                        current['id'], current['run_id'])
                active = gpu_processes()
                if active:
                    raise RuntimeError('다른 GPU 프로세스가 있어 경합 없이 차단합니다: ' + ','.join(map(str, active)))
                current['status'], current['error'] = 'training', None
                state['status'], state['current_condition_id'] = 'training', current['id']
                write_state(root, state)
                run_child(root, state, current, training_argv(root, plan['python'], plan['project'],
                    current['id'], current['run_id'], resume=resume), 'training')
                current['status'], state['status'] = 'evaluating', 'evaluating'
                write_state(root, state)
                if gpu_processes():
                    raise RuntimeError('학습 뒤 GPU 프로세스가 남아 있어 평가를 시작하지 않습니다.')
                run_child(root, state, current, evaluation_argv(root, plan['python'], plan['project'],
                    current['id'], current['run_id']), 'evaluation')
                current['completion'] = completed_condition_gate(root, current)
                current['status'] = 'completed'
                write_state(root, state)
            state['status'], state['current_condition_id'] = 'completed', None
            write_state(root, state)
        except BaseException as error:
            state['status'] = 'blocked' if isinstance(error, (KeyboardInterrupt, ValueError, FileNotFoundError, OSError)) else 'failed'
            state['error'] = type(error).__name__ + ': ' + str(error)
            if current is not None and current.get('status') in ('training', 'evaluating'):
                current['status'], current['error'] = 'failed', state['error']
            write_state(root, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--baseline-run-id')
    parser.add_argument('--baseline-pid', type=int)
    parser.add_argument('--baseline-created-utc')
    parser.add_argument('--resume-condition', choices=[row['id'] for row in condition_plan()])
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if root.name != '10차' or root.parent.name != '흉부 엑스레이 환자 재식별 실험' or root.drive.upper() != 'E:':
        parser.error('실제 E: 프로젝트10차 전용 큐입니다.')
    if not 5 <= args.poll_seconds <= 60:
        parser.error('조회 주기는5~60초입니다.')
    if args.prepare and (not args.baseline_run_id or not args.baseline_pid or not args.baseline_created_utc):
        parser.error('준비에는 기존 run UUID/PID/생성시각이 모두 필요합니다.')
    plain_path(root)
    if args.prepare:
        print(json.dumps(prepare(root, args), ensure_ascii=False))
    else:
        execute(root, args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
