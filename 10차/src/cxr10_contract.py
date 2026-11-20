"""강도별 새 모델·자기 조건 재개의 계약. GPU 패키지를 가져오지 않습니다."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid


LEVELS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
PROTOCOL_NAME = 'independent_strength_training_both_sides_v1'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def condition_plan():
    rows = [{'id': 'original_000', 'mode': 'original', 'strength': 0,
        'identity_transform': True, 'operator_mode': 'original', 'operator_strength': 0,
        'units': 'none', 'training_sides': 'all_training_and_validation_images',
        'evaluation_sides': 'transformed_query_and_transformed_gallery',
        'original_image_modified': False}]
    for mode in ('blur', 'mosaic'):
        for strength in LEVELS:
            identity = strength == 0 or (mode == 'mosaic' and strength == 1)
            rows.append({'id': f'{mode}_{strength:03d}', 'mode': mode, 'strength': strength,
                'identity_transform': identity, 'operator_mode': 'original' if identity else mode,
                'operator_strength': 0 if identity else strength,
                'units': 'source_pixel_gaussian_standard_deviation' if mode == 'blur' else 'source_pixel_tile_size',
                'training_sides': 'all_training_and_validation_images',
                'evaluation_sides': 'transformed_query_and_transformed_gallery',
                'original_image_modified': False})
    return rows


def condition(condition_id):
    if not isinstance(condition_id, str):
        raise ValueError('조건 ID는 고정 허용목록의 문자열이어야 합니다.')
    for row in condition_plan():
        if row['id'] == condition_id:
            return row
    raise ValueError('허용되지 않은 독립학습 조건입니다.')


def condition_fingerprint(condition_id):
    return canonical_digest({'protocol': PROTOCOL_NAME, 'condition': condition(condition_id),
        'seed': 3, 'initialization': 'fresh_imagenet_seed3',
        'cross_condition_state_inherited': False})


def valid_run_id(run_id):
    try:
        parsed = uuid.UUID(run_id)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError('실행 ID는 정규 UUID여야 합니다.') from error
    if str(parsed) != run_id:
        raise ValueError('실행 UUID는 소문자 정규 표현만 허용합니다.')
    return run_id


def run_relative(run_id):
    return 'work/runs/' + valid_run_id(run_id) + '/training'


def training_identity(condition_id, run_id):
    return {'condition_id': condition(condition_id)['id'],
        'condition_fingerprint': condition_fingerprint(condition_id),
        'condition': condition(condition_id), 'run_id': valid_run_id(run_id),
        'initialization_mode': 'fresh_imagenet_seed3',
        'cross_condition_state_inherited': False, 'experiment_protocol': PROTOCOL_NAME}


def validate_training_identity(saved, condition_id, run_id):
    for key, expected in training_identity(condition_id, run_id).items():
        if saved.get(key) != expected:
            raise ValueError('다른 조건·실행·프로토콜의 상태를 승계할 수 없습니다: ' + key)


def protocol_fingerprint(condition_id, run_id, complete_protocol):
    """호출자가 조사한 전체 설정·소스·목록·환경을 조건과 실행에 결합합니다."""
    return canonical_digest({'independent_training': training_identity(condition_id, run_id),
        'complete_protocol': complete_protocol})


def plain_path(path):
    for part in (Path(path), *Path(path).parents):
        if part.exists():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
                raise ValueError('링크·재분석 경로는 학습 입출력으로 허용하지 않습니다.')


def assert_run_scope(root, output, project):
    root, output, project = Path(root), Path(output), Path(project)
    for path in (root, output, project):
        if '..' in path.parts:
            raise ValueError('상위 경로 이동을 허용하지 않습니다.')
        plain_path(path)
    if root.name != '10차' or root.parent.name != '흉부 엑스레이 환자 재식별 실험' or root.drive.upper() != 'E:':
        raise ValueError('이 학습 사본은 해당 프로젝트 E: 10차에서만 실행합니다.')
    relative = output.resolve().relative_to(root.resolve())
    if len(relative.parts) != 4 or relative.parts[:2] != ('work', 'runs') or relative.parts[3] != 'training':
        raise ValueError('출력은10차/work/runs/<UUID>/training이어야 합니다.')
    valid_run_id(relative.parts[2])
    if project.resolve() != (root.parent / '연구 자료').resolve():
        raise ValueError('입력 연구 자료 경로가 프로젝트 공통 원본과 다릅니다.')


def atomic_json(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path = Path(path)
    plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.1 * (attempt + 1))
    finally:
        if temp.exists():
            temp.unlink()


def read_json(path, limit=4 * 1024 * 1024):
    path = Path(path)
    plain_path(path)
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError('상태 파일 크기 상한을 초과했습니다.')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('JSON 중복 키를 허용하지 않습니다.')
            value[key] = item
        return value
    return json.loads(data.decode('utf-8-sig', 'strict'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('비유한 JSON 수치')))


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def initial_queue_state(queue_id, baseline, *, start_mode='wait_baseline'):
    return {'schema_version': 1, 'queue_id': valid_run_id(queue_id),
        'status': 'ready' if start_mode == 'fresh_all_21' else 'waiting_baseline',
        'start_mode': start_mode,
        'updated_at': utc_now(), 'current_condition_id': None, 'baseline': baseline,
        'protocol': PROTOCOL_NAME, 'conditions': [{**row, 'status': 'pending', 'run_id': None,
            'training_subdir': None, 'condition_fingerprint': condition_fingerprint(row['id']),
            'initialization': 'fresh_imagenet_seed3', 'process': None, 'error': None}
            for row in condition_plan()]}


def training_argv(root, python, project, condition_id, run_id, *, resume=False):
    condition(condition_id)
    root = Path(root)
    argv = [str(python), '-B', '-X', 'utf8', '-u', str(root / 'src/train_from_scratch.py'),
        '--project', str(project), '--output', str(root / run_relative(run_id)),
        '--condition-id', condition_id, '--tasks', 'both', '--seed', '3',
        '--verification-workers', '16', '--retrieval-workers', '4',
        '--activation-checkpointing',
        '--imagenet-cache', str(root / 'work/imagenet_cache')]
    if resume:
        argv.append('--resume')
    return argv


def evaluation_argv(root, python, project, condition_id, run_id):
    condition(condition_id)
    return [str(python), '-B', '-X', 'utf8', '-u', str(Path(root) / 'src/cxr10_evaluate.py'),
        '--project', str(project), '--output', str(Path(root) / run_relative(run_id)),
        '--condition-id', condition_id, '--tasks', 'both', '--workers', '4',
        '--retrieval-batch', '4', '--bootstrap', '10000']
