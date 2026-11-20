"""정확히 고정한 공식 ImageNet legacy 파일만 안전한 저장 형식으로 변환합니다."""

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torchvision.models import resnet50

from cxr_reid.download import download_with_resume
from .state import atomic_json, digest


SOURCE_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
SOURCE_SHA256 = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"
SOURCE_BYTES = 102502400


def validate_legacy_bytes(blob):
    if len(blob) != SOURCE_BYTES:
        raise ValueError("공식 ImageNet legacy 파일의 고정 크기가 다릅니다.")
    if hashlib.sha256(blob).hexdigest() != SOURCE_SHA256:
        raise ValueError("공식 ImageNet legacy 파일의 고정 SHA256가 다릅니다.")


def validate_state(state):
    if not isinstance(state, (dict, OrderedDict)):
        raise ValueError("ImageNet 변환 결과가 state_dict가 아닙니다.")
    expected = resnet50(weights=None).state_dict()
    # 구버전 BatchNorm에는 num_batches_tracked가 없으며 PyTorch의 정식 로더가 0을 보충합니다.
    keys_without_counters = {key for key in expected if not key.endswith("num_batches_tracked")}
    if set(state) not in (set(expected), keys_without_counters):
        raise ValueError("공식 ImageNet 모델의 키 집합이 다릅니다.")
    for name, tensor in state.items():
        if type(tensor) is not torch.Tensor or tensor.device.type != "cpu":
            raise ValueError("공식 ImageNet state_dict에 일반 CPU tensor가 아닌 값이 있습니다: "
                             f"key={name}, type={type(tensor).__module__}.{type(tensor).__name__}, "
                             f"device={getattr(tensor, 'device', None)}, "
                             f"shape={getattr(tensor, 'shape', None)}, dtype={getattr(tensor, 'dtype', None)}")
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError("공식 ImageNet tensor의 shape/dtype가 다릅니다.")
        if not torch.isfinite(tensor).all():
            raise ValueError("공식 ImageNet tensor에 비유한 값이 있습니다.")


def normalize_legacy_state(state):
    """고정 공식 파일의 Parameter 저장 표현만 순수 CPU Tensor로 바꿉니다."""
    if not isinstance(state, (dict, OrderedDict)):
        raise ValueError("공식 legacy ImageNet 반환 값이 state_dict가 아닙니다.")
    normalized = OrderedDict()
    for name, tensor in state.items():
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.device.type != "cpu":
            raise ValueError("공식 legacy ImageNet에 허용하지 않은 tensor 하위 자료형 또는 장치가 있습니다.")
        normalized[name] = tensor.detach().cpu()
        if type(normalized[name]) is not torch.Tensor or not torch.equal(tensor, normalized[name]):
            raise ValueError("ImageNet Parameter의 저장 표현 정규화에서 값이 변경됐습니다.")
    validate_state(normalized)
    return normalized


def convert_legacy_bytes(blob):
    """검증한 공식 바이트만 작업자 전용 임시 핸들에서 변환합니다."""
    validate_legacy_bytes(blob)
    if __name__ != "__main__":
        raise RuntimeError("실제 legacy 역직렬화는 별도 변환 subprocess에서만 허용합니다.")
    # 구형 tar 로더는 fileno()를 요구하므로 BytesIO 대신 보호된 임시 핸들을 사용합니다.
    # 부모가 승인한 공식 단일 해시만 도달하며 외부 파일 경로·해시·옵션은 받지 않습니다.
    work = Path(__file__).resolve().parents[2] / "work" / "imagenet_conversion"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="worker-", dir=work) as directory:
        # torch의 구형 tar 로더가 만드는 내부 추출 임시 폴더도 이 작업 경계에 둡니다.
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = directory
        try:
            with tempfile.TemporaryFile(mode="w+b", prefix="verified-", dir=directory) as handle:
                handle.write(blob)
                handle.flush()
                handle.seek(0)
                validate_legacy_bytes(handle.read())
                handle.seek(0)
                original_state = torch.load(handle, map_location="cpu", weights_only=False)
        finally:
            tempfile.tempdir = previous_tempdir
    state = normalize_legacy_state(original_state)
    stream = io.BytesIO()
    torch.save(state, stream)
    safe = stream.getvalue()
    reloaded = torch.load(io.BytesIO(safe), map_location="cpu", weights_only=True)
    validate_state(reloaded)
    if (list(original_state) != list(state) or list(state) != list(reloaded)
            or not all(torch.equal(original_state[key], state[key])
                       and torch.equal(state[key], reloaded[key]) for key in state)):
        raise ValueError("ImageNet 형식 변환 전후 tensor가 정확하게 일치하지 않습니다.")
    return safe, {"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                  "source_bytes": SOURCE_BYTES, "converted_sha256": hashlib.sha256(safe).hexdigest(),
                  "converted_bytes": len(safe), "tensor_count": len(state),
                  "legacy_parameter_count": sum(type(value) is torch.nn.Parameter for value in original_state.values()),
                  "normalization": "정확한 nn.Parameter만 detach().cpu() 순수 Tensor로 저장 표현 변환; 수치 변경 없음",
                  "all_tensors_equal": True, "keys_shapes_dtypes_finite_checked": True,
                  "legacy_load_boundary": "고정 SHA/크기·동일 바이트와 임시 핸들 재검증·별도 subprocess 한정"}


def ensure_safe_initialization(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / "resnet50-19c8e357.pth"
    safe = cache / "resnet50-19c8e357.safe.pt"
    receipt = cache / "resnet50-19c8e357.safe.json"
    download_with_resume(SOURCE_URL, source, digest=SOURCE_SHA256)
    if safe.exists() or receipt.exists():
        if not safe.is_file() or not receipt.is_file():
            raise ValueError("ImageNet 변환 캐시 또는 영수증이 누락됐습니다. 기존 파일을 보존합니다.")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if (record.get("source_sha256") != SOURCE_SHA256 or record.get("source_bytes") != SOURCE_BYTES
                or record.get("source_url") != SOURCE_URL or not record.get("all_tensors_equal")
                or record.get("converted_sha256") != digest(safe)):
            raise ValueError("ImageNet 변환 캐시의 출처·바이트 해시가 다릅니다. 기존 파일을 보존합니다.")
    else:
        blob = source.read_bytes()
        validate_legacy_bytes(blob)
        command = [sys.executable, "-B", "-X", "utf8", "-m", "training.imagenet"]
        result = subprocess.run(command, input=blob, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=Path(__file__).resolve().parents[1], timeout=180, check=False)
        if result.returncode:
            raise RuntimeError("고정 ImageNet 변환 subprocess 실패: " + result.stderr.decode("utf-8", errors="replace")[-5000:])
        marker = b"IMAGENET_CONVERSION_PROVENANCE "
        lines = result.stderr.split(b"\n")
        records = [json.loads(line[len(marker):]) for line in lines if line.startswith(marker)]
        if len(records) != 1:
            raise ValueError("ImageNet 변환 subprocess의 영수증이 없습니다.")
        record = records[0]
        if record.get("converted_sha256") != hashlib.sha256(result.stdout).hexdigest():
            raise ValueError("ImageNet 변환 subprocess 출력이 영수증과 다릅니다.")
        validate_state(torch.load(io.BytesIO(result.stdout), map_location="cpu", weights_only=True))
        # 중단으로 남은 임시 파일을 덮어쓰거나 고정 임시 이름을 다시 사용하지 않습니다.
        with tempfile.NamedTemporaryFile(mode="wb", prefix=safe.name + ".", suffix=".tmp",
                                         dir=cache, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(result.stdout)
        try:
            temporary.replace(safe)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(receipt, record)
    state = torch.load(safe, map_location="cpu", weights_only=True)
    validate_state(state)
    return state, {**record, "safe_file": str(safe), "sha256": digest(safe),
                   "kind": "ImageNet 사전학습 backbone; 공개 CXR 최종 모델 아님"}


def main():
    if len(sys.argv) != 1:
        raise ValueError("고정 ImageNet 변환 작업자는 파일 경로나 옵션을 받지 않습니다.")
    torch.set_num_threads(2)
    blob = sys.stdin.buffer.read(SOURCE_BYTES + 1)
    safe, record = convert_legacy_bytes(blob)
    sys.stderr.write("IMAGENET_CONVERSION_PROVENANCE " + json.dumps(record) + "\n")
    sys.stderr.flush()
    sys.stdout.buffer.write(safe)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
