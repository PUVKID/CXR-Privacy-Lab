"""종료증거 누락·프로토콜 변경·조건 섞임과 순차 실행의 실패 경로를 검사합니다."""

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cxr10_contract as c
import cxr10_queue as q


def condition_outputs(root, row, *, evaluation=True, exits=True):
    output = root / c.run_relative(row['run_id'])
    output.mkdir(parents=True, exist_ok=True)
    identity = c.training_identity(row['id'], row['run_id'])
    binding = 'a' * 64
    config = {**identity, 'binding': binding, 'protocol_fingerprint': binding}
    artifacts = {**config, 'published_cxr_initialization': False}
    hashes = {}
    for task, phase in (('verification', 'verification'), ('retrieval', 'retrieval_phase2')):
        folder = output / phase
        folder.mkdir(exist_ok=True)
        weight = folder / 'selected_weights.pth'
        weight.write_bytes(task.encode('ascii'))
        hashes[task] = hashlib.sha256(task.encode('ascii')).hexdigest()
        artifacts[task + '_checkpoint'] = str(weight)
        artifacts[task + '_sha256'] = hashes[task]
    c.atomic_json(output / 'training_config.json', config)
    c.atomic_json(output / 'artifacts.json', artifacts)
    if exits:
        c.atomic_json(output / 'training_exit.json', {'exit_code': 0, 'condition_id': row['id'], 'run_id': row['run_id']})
    if evaluation:
        result = {**identity, 'training_binding': binding, 'completed': True}
        for dataset in ('NIH', 'COVID19'):
            result[dataset] = {'verification': {'auc': .8, 'accuracy': .7},
                'retrieval': {'precision_at_1': .5, 'map_at_r': .4, 'r_precision': .3},
                'verification_sha256': hashes['verification'], 'retrieval_sha256': hashes['retrieval']}
        c.atomic_json(output / 'evaluation_results.json', result)
        if exits:
            c.atomic_json(output / 'evaluation_exit.json', {'exit_code': 0, 'condition_id': row['id'], 'run_id': row['run_id']})


class QueueSafetyTests(unittest.TestCase):
    def test_complete_protocol_rejects_lr_epoch_resolution_manifest_or_source_change(self):
        contract = {'protocol': {'lr': .0001, 'epochs': 100, 'resolution': 256},
            'sources': {'training/data.py': 'a' * 64}, 'manifests': {'train.csv': 'b' * 64}}
        run = '11111111-1111-4111-8111-111111111111'
        before = c.protocol_fingerprint('blur_004', run, contract)
        for group, key, value in [('protocol', 'lr', .01), ('protocol', 'epochs', 99),
                ('protocol', 'resolution', 128), ('sources', 'training/data.py', 'c' * 64),
                ('manifests', 'train.csv', 'd' * 64)]:
            changed = copy.deepcopy(contract)
            changed[group][key] = value
            with self.subTest(group=group, key=key):
                self.assertNotEqual(before, c.protocol_fingerprint('blur_004', run, changed))

    def test_missing_actual_exit_cannot_be_reused_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = {'pid': 77, 'creation_filetime': '123', 'command_line': 'same'}
            plan = {'queue_id': str(uuid.uuid4()), 'baseline': {'process': process}}
            c.atomic_json(root / 'work/queue/baseline_exit.private.json',
                {'queue_id': plan['queue_id'], 'process': process})
            with self.assertRaises(ValueError):
                q.wait_baseline(root, plan, {}, 5)

    def test_artifact_files_without_successful_child_exits_do_not_complete_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'id': 'blur_001', 'run_id': str(uuid.uuid4())}
            condition_outputs(root, row, exits=False)
            with self.assertRaises((ValueError, FileNotFoundError)):
                q.completed_condition_gate(root, row)

    def test_evaluation_from_different_condition_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'id': 'blur_001', 'run_id': str(uuid.uuid4())}
            condition_outputs(root, row)
            path = root / c.run_relative(row['run_id']) / 'evaluation_results.json'
            result = c.read_json(path)
            result['condition_id'] = 'blur_002'
            c.atomic_json(path, result)
            with self.assertRaises(ValueError):
                q.completed_condition_gate(root, row)

    def test_failed_evaluation_exit_is_rejected_even_when_metrics_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'id': 'mosaic_001', 'run_id': str(uuid.uuid4())}
            condition_outputs(root, row)
            c.atomic_json(root / c.run_relative(row['run_id']) / 'evaluation_exit.json',
                {'condition_id': row['id'], 'run_id': row['run_id'], 'exit_code': 1})
            with self.assertRaises((ValueError, RuntimeError)):
                q.completed_condition_gate(root, row)

    def test_duplicate_run_or_changed_condition_in_state_is_rejected(self):
        plan = {'queue_id': str(uuid.uuid4())}
        state = c.initial_queue_state(plan['queue_id'], {})
        run = str(uuid.uuid4())
        for row in state['conditions'][:2]:
            row.update(run_id=run, training_subdir=c.run_relative(run), status='completed')
        with self.assertRaises(ValueError):
            q.validate_state(state, plan)
        state = c.initial_queue_state(plan['queue_id'], {})
        state['conditions'][0]['operator_mode'] = 'blur'
        with self.assertRaises(ValueError):
            q.validate_state(state, plan)

    def test_plan_cannot_choose_other_python_or_dataset(self):
        self.assertTrue(hasattr(q, 'validate_plan'), '고정 실행기/원본 경로의 계획 검사가 필요합니다.')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / '10차'
            plan = {'schema_version': 1, 'queue_id': str(uuid.uuid4()), 'protocol': c.PROTOCOL_NAME,
                'python': str(root.parent / '실행 환경/Scripts/python.exe'),
                'project': str(root.parent / '연구 자료'), 'source_fingerprint': 'a' * 64,
                'baseline': {'run_id': str(uuid.uuid4()), 'process': {'pid': 77,
                    'creation_filetime': '134396201648233456', 'command_line': 'same'}}}
            q.validate_plan(root, plan)
            for key, value in [('python', 'other.py'), ('project', 'other-data')]:
                changed = copy.deepcopy(plan)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    q.validate_plan(root, changed)

    def run_fixture(self, root, *, fail_condition=None):
        (root / 'src').mkdir(parents=True)
        (root / 'src/fixture.py').write_text('# 합성 검사\n', encoding='utf-8')
        plan = {'schema_version': 1, 'queue_id': str(uuid.uuid4()), 'protocol': c.PROTOCOL_NAME,
            'python': str(root.parent / '실행 환경/Scripts/python.exe'), 'project': str(root.parent / '연구 자료'),
            'source_fingerprint': q.source_fingerprint(root), 'baseline': {'run_id': str(uuid.uuid4()),
            'process': {'pid': 77, 'creation_filetime': '123', 'command_line': 'same'}}}
        c.atomic_json(root / 'work/queue/plan.private.json', plan)
        q.write_state(root, c.initial_queue_state(plan['queue_id'], plan['baseline']))
        seen = []
        def child(root, state, row, argv, stage):
            seen.append((row['id'], row['run_id'], stage))
            if row['id'] == fail_condition:
                raise RuntimeError('합성 child exit1')
            condition_outputs(root, row, evaluation=stage == 'evaluation')
        fake_handle = SimpleNamespace(identity={'pid': 88, 'creation_filetime': '222', 'command_line': 'queue'})
        args = SimpleNamespace(poll_seconds=5, resume_condition=None)
        with patch.object(q, 'prevent_sleep', return_value=nullcontext()), \
             patch.object(q, 'ProcessHandle', return_value=nullcontext(fake_handle)), \
             patch.object(q, 'wait_baseline'), patch.object(q, 'baseline_artifact_gate', return_value={'completed_task_conditions': 84}), \
             patch.object(q, 'gpu_processes', return_value=[]), patch.object(q, 'run_child', side_effect=child):
            try:
                q.execute(root, args)
            except RuntimeError:
                if fail_condition is None:
                    raise
        return c.read_json(root / 'work/queue/state.json'), seen

    def test_queue_runs_each_condition_training_then_evaluation_in_separate_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            state, seen = self.run_fixture(Path(directory))
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(len(seen), 40)
            self.assertEqual([item[2] for item in seen], ['training', 'evaluation'] * 20)
            self.assertEqual(len({item[1] for item in seen}), 20)
            self.assertEqual(len({item[0] for item in seen}), 20)

    def test_queue_failure_never_starts_later_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            state, seen = self.run_fixture(Path(directory), fail_condition='blur_002')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual([item[0] for item in seen], ['blur_000', 'blur_000', 'blur_001', 'blur_001', 'blur_002'])
            self.assertTrue(all(row['status'] == 'pending' for row in state['conditions'][3:]))


if __name__ == '__main__':
    unittest.main()
