"""5차의 데이터 검증·새 모델 학습·새 가중치 평가를 순서대로 관리합니다."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "연구 자료"
CONTROL = HERE / "work/control"
PYTHON = HERE.parent / "실행 환경/Scripts/python.exe"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def stop_child(child):
    """이 실행이 시작한 프로세스와 후손만 종료합니다."""
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        child.terminate()
    child.wait(timeout=30)


def acquire_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise
    return handle


def main(prepare_only=False, validated_snapshot=None):
    CONTROL.mkdir(parents=True, exist_ok=True)
    try:
        # 이전 차수 관리자와도 같은 데이터 준비를 동시에 실행하지 않습니다.
        lock = acquire_lock(BASE / "control/run.lock")
    except OSError:
        print("[실행 거절] 데이터 경로를 사용하는 실험이 있습니다.", flush=True)
        return 2
    run_id = str(uuid.uuid4())
    output = HERE / "work/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    state = {"run_id": run_id, "pid": os.getpid(), "status": "running",
             "phase": "학습 실행환경 확인", "revision": 5,
             "mode": "데이터 준비 전용" if prepare_only else "새 모델 재학습 후 평가", "output": str(output),
             "started_at": timestamp(), "finished_at": None, "exit_code": None, "error": None}
    children = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               MPLBACKEND="Agg", TORCH_HOME=str(HERE / "work/cache/torch"))
    log = (CONTROL / "experiment.log").open("a", encoding="utf-8", buffering=1)
    owned_log = (output / "experiment.log").open("x", encoding="utf-8", buffering=1)

    def update(**values):
        state.update(values)
        atomic_json(CONTROL / "run.json", state)
        atomic_json(output / "run.json", state)

    def message(value):
        log.write(value + "\n")
        owned_log.write(value + "\n")
        print(value, flush=True)

    def cancelled():
        try:
            return (CONTROL / "stop.request").read_text(encoding="utf-8").strip() == run_id
        except FileNotFoundError:
            return False

    commands = []

    def execute(argv, phase):
        if cancelled():
            raise InterruptedError("사용자가 실험 중단을 요청했습니다.")
        update(phase=phase)
        command = [str(x) for x in argv]
        commands.append({"phase": phase, "argv": command, "cwd": str(HERE / "실험 코드"), "at": timestamp()})
        atomic_json(output / "commands.json", commands)
        message("[실행] " + json.dumps(command, ensure_ascii=False))
        child = subprocess.Popen(command, cwd=HERE / "실험 코드", env=env,
                                 stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        children.append(child)
        while child.poll() is None:
            if cancelled():
                raise InterruptedError("사용자가 실험 중단을 요청했습니다.")
            time.sleep(1)
        if cancelled():
            raise InterruptedError("사용자가 실험 중단을 요청했습니다.")
        if child.returncode:
            raise RuntimeError(f"{phase}: 종료 코드 {child.returncode}. 원본 실행 기록을 확인하세요.")

    try:
        update()
        message(f"\n[새 실행] {timestamp()} / {run_id} / 5차")
        message("[학습 방식] 공개 CXR 최종 가중치 없이 새 학습. 논문상 ImageNet 초기화와 구분합니다.")
        if prepare_only:
            message("[준비 전용] 이번 실행에서는 데이터 준비와 검증만 수행합니다. 학습을 시작하지 않습니다.")
        source = {}
        for folder in [HERE / "실험 코드", HERE / "화면"]:
            for file in sorted(folder.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts:
                    source[file.relative_to(HERE).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        atomic_json(output / "source_manifest.json", source)
        execute([PYTHON, "-B", "-X", "utf8", "-c",
                 "import torch,torchvision,training.engine; assert torch.cuda.is_available(); "
                 "x=torch.ones(4,device='cuda'); assert x.sum().item()==4; "
                 "print('[CUDA 검증]',torch.__version__,torch.cuda.get_device_name(0),flush=True)"],
                "학습 실행환경과 CUDA 실제 연산 확인")
        if validated_snapshot is None:
            previous_reports = set((BASE / "validation_runs").glob("*/report.json"))
            execute([PYTHON, "-B", "-X", "utf8", "-u", HERE / "실험 코드/download_datasets.py", "--project", BASE,
                     "--reuse-nih-pixel-manifest", BASE / "validation_runs/원본_전수검사/nih_pixels.private.json"],
                    "데이터 준비·영상 전수 검증·환자 분리 확인")
            new_reports = set((BASE / "validation_runs").glob("*/report.json")) - previous_reports
            if len(new_reports) != 1:
                raise RuntimeError("이번 준비 실행의 검증 보고서를 하나로 특정할 수 없습니다.")
            validated_snapshot = new_reports.pop().parent.resolve(strict=True)
            update(validation_snapshot=str(validated_snapshot))
        else:
            validated_snapshot = Path(validated_snapshot).resolve(strict=True)
            if not validated_snapshot.is_relative_to((BASE / "validation_runs").resolve()):
                raise ValueError("현재 연구 경로 안의 검증 스냅샷만 사용할 수 있습니다.")
            update(validation_snapshot=str(validated_snapshot))
            execute([PYTHON, "-B", "-X", "utf8", "-u", "-c",
                     "from pathlib import Path; import sys; from cxr_reid.dataset_validation import validate_prepared_snapshot; "
                     "validate_prepared_snapshot(Path(sys.argv[1]),Path(sys.argv[2])); print('[데이터 검증 완료] 모든 원영상 SHA256 재확인',flush=True)",
                     BASE, validated_snapshot], "검증 스냅샷과 현재 원영상 SHA256 전수 재확인")
        if prepare_only:
            update(status="completed", phase="데이터 준비와 검증 완료 · 학습은 아직 시작하지 않음", exit_code=0, finished_at=timestamp())
            message("[준비 완료] 데이터 준비 전용 실행이며 모델 학습은 수행하지 않았습니다.")
            return 0
        execute([PYTHON, "-B", "-X", "utf8", "-u", HERE / "실험 코드/train_from_scratch.py",
                 "--project", BASE, "--output", output / "training", "--seed", "3", "--tasks", "both",
                 "--validated-snapshot", validated_snapshot, "--activation-checkpointing", "--num-workers", "0"],
                "검증 모델·검색 모델 처음부터 학습")
        artifacts = json.loads((output / "training/artifacts.json").read_text(encoding="utf-8"))
        checkpoints = []
        for key in ["verification_checkpoint", "retrieval_checkpoint"]:
            candidate = Path(artifacts[key]).resolve(strict=True)
            if not candidate.is_relative_to((output / "training").resolve()) or not candidate.is_file():
                raise ValueError("현재 실행이 생성한 학습 체크포인트가 아닙니다.")
            checkpoints.append(candidate)
        evaluation = [PYTHON, "-B", "-X", "utf8", "-u", HERE / "실험 코드/main.py", "--project", BASE,
                 "--stage", "all", "--skip-download", "--run-id", run_id,
                 "--verification-checkpoint", checkpoints[0], "--retrieval-checkpoint", checkpoints[1]]
        if validated_snapshot is not None:
            evaluation.extend(["--validated-snapshot", validated_snapshot])
        execute(evaluation, "새 학습 가중치로 1→2→3단계 평가 실행")
        update(status="completed", phase="새 모델 학습과 모든 평가 완료", exit_code=0, finished_at=timestamp())
        message("[완료] 새 학습과 원본·블러·모자이크 평가가 정상 종료했습니다.")
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
        owned_log.close()
        lock.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="데이터 준비와 검증만 수행합니다")
    parser.add_argument("--validated-snapshot", type=Path, help="원영상 SHA256 전수 재확인으로 기존 검증의 디코드 결과를 재사용합니다")
    args = parser.parse_args()
    sys.exit(main(prepare_only=args.prepare_only, validated_snapshot=args.validated_snapshot))
