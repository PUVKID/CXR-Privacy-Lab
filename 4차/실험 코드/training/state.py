"""가중치·최적화 상태·난수 상태를 원자적으로 저장하고 입력 결합을 검사합니다."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    random.setstate(value["python"])
    np_state = value["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(value["torch"])
    if value["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def save_checkpoint(path, model, optimizer, *, metadata, scheduler=None, scaler=None, memory=None):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
               "metadata": metadata, "scheduler": scheduler.state_dict() if scheduler else None,
               "scaler": scaler.state_dict() if scaler else None,
               "memory": memory.state_dict() if memory else None}
    atomic_torch_save(payload, path)


def load_checkpoint(path, model, optimizer, *, binding, scheduler=None, scaler=None, memory=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"]["binding"] != binding:
        raise ValueError("재개할 체크포인트의 소스·설정·데이터 분할이 현재 실행과 다릅니다.")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for obj, key in ((scheduler, "scheduler"), (scaler, "scaler"), (memory, "memory")):
        if obj is not None and payload[key] is not None:
            obj.load_state_dict(payload[key])
    restore_rng(payload["rng"])
    return payload["metadata"]


def progress(output, **value):
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(Path(output) / "progress.json", value)
    # JSONL은 실제 LF 하나로만 레코드를 나눕니다.
    with open(Path(output) / "events.jsonl", "a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
