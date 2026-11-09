"""독립 실행·중단·환경 준비와 원본 3단계 평가를 관리합니다."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "research"
CONTROL = BASE / "control"
PYTHON = HERE.parent / "runtime/Scripts/python.exe"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def stop_child(child):
    """이 실행에서 직접 시작한 프로세스 트리만 종료합니다."""
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        child.terminate()
    child.wait(timeout=30)


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / "run.lock").open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[실행 거절] 이미 실행 중인 실험이 있습니다.", flush=True)
        return 2

    run_id = str(uuid.uuid4())
    state = {"run_id": run_id, "pid": os.getpid(), "status": "running",
             "phase": "환경 및 데이터 준비", "started_at": timestamp(),
             "finished_at": None, "exit_code": None, "error": None}
    atomic_json(CONTROL / "run.json", state)
    children = []
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               MPLBACKEND="Agg")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def update(**values):
        state.update(values)
        atomic_json(CONTROL / "run.json", state)

    def cancelled():
        stop_file = CONTROL / "stop.request"
        try:
            return stop_file.read_text(encoding="utf-8").strip() == run_id
        except FileNotFoundError:
            return False

    log = (CONTROL / "experiment.log").open("a", encoding="utf-8", buffering=1)
    package_log = (CONTROL / "environment.log").open("a", encoding="utf-8", buffering=1)

    def message(value):
        log.write(f"{value}\n")
        log.flush()

    def launch(argv, output=log):
        message("[실행] " + json.dumps([str(x) for x in argv], ensure_ascii=False))
        child = subprocess.Popen([str(x) for x in argv], cwd=HERE / "experiment", env=env,
                                 stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
        children.append(child)
        return child

    def wait(child, phase):
        update(phase=phase)
        while child.poll() is None:
            if cancelled():
                raise InterruptedError("사용자가 실험 중단을 요청했습니다.")
            time.sleep(1)
        if child.returncode:
            raise RuntimeError(f"{phase}: 종료 코드 {child.returncode}. 실행 기록을 확인하세요.")

    try:
        message(f"\n[새 실행] {timestamp()} / {run_id}")
        message("[평가 방식] 공개 최종 가중치 평가 재현이며 새 모델 학습이 아닙니다.")
        requirements = HERE / "experiment/requirements.txt"
        signature = hashlib.sha256(requirements.read_bytes() + b"torch2.7.1_tv0.22.1_cu126_py312").hexdigest()
        marker = CONTROL / "environment_ready.json"
        ready = marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("signature") == signature
        # 공식 데이터 다운로드는 표준 라이브러리만 사용하므로 환경 설치와 병행합니다.
        download = launch([PYTHON, "-u", HERE / "experiment/download_datasets.py", "--project", BASE])
        if not ready:
            torch_install = launch([PYTHON, "-m", "pip", "install", "torch==2.7.1", "torchvision==0.22.1",
                                    "--index-url", "https://download.pytorch.org/whl/cu126"], package_log)
            wait(torch_install, "CUDA 실행환경 설치 · 데이터 다운로드 병행")
            packages = launch([PYTHON, "-m", "pip", "install", "-r", requirements], package_log)
            wait(packages, "실험 의존성 설치 · 데이터 다운로드 병행")
            check = launch([PYTHON, "-c", "import torch, torchvision, numpy, sklearn, pandas, PIL; "
                            "assert torch.cuda.is_available(), 'CUDA 사용 불가'; "
                            "x=torch.ones(4,device='cuda'); assert x.sum().item()==4; "
                            "print('[CUDA 검증]',torch.__version__,torch.cuda.get_device_name(0),flush=True)"])
            wait(check, "CUDA 실제 연산 검사")
            atomic_json(marker, {"signature": signature, "verified_at": timestamp(), "python": sys.version})
        wait(download, "NIH·COVID-19 다운로드 및 원본 검증")
        evaluation = launch([PYTHON, "-u", HERE / "experiment/main.py", "--project", BASE,
                             "--stage", "all", "--skip-download"])
        wait(evaluation, "1→2→3단계 평가 실행")
        update(status="completed", phase="모든 단계 평가 완료", exit_code=0, finished_at=timestamp())
        message("[완료] 원본·블러·모자이크 평가 프로세스가 정상 종료했습니다.")
        return 0
    except InterruptedError as error:
        for child in children:
            stop_child(child)
        update(status="stopped", phase="사용자 중단", error=str(error), exit_code=130, finished_at=timestamp())
        message("[중단] " + str(error))
        return 130
    except Exception as error:
        for child in children:
            stop_child(child)
        update(status="failed", phase="실행 오류", error=str(error), exit_code=1, finished_at=timestamp())
        message("[실패] " + repr(error))
        return 1
    finally:
        log.close()
        package_log.close()
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
