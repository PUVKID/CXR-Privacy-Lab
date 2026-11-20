"""조건 혼입·잘못된 완료 판정이 GPU 학습으로 이어지는 오류를 검출합니다."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

SOURCE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SOURCE))


class ConditionContractTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('cxr10_contract'), '독립 조건 계약 구현이 필요합니다.')
        import cxr10_contract
        return cxr10_contract

    def test_twenty_conditions_have_separate_identities_even_when_operator_is_identity(self):
        c = self.module()
        rows = c.condition_plan()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row['id'] for row in rows}), 20)
        self.assertEqual([row['id'] for row in rows if row['identity_transform']],
                         ['blur_000', 'mosaic_000', 'mosaic_001'])
        self.assertEqual(len({c.condition_fingerprint(row['id']) for row in rows}), 20)

    def test_unrecognized_condition_cannot_select_a_file_or_model(self):
        c = self.module()
        for value in ('../blur_001', 'original_000', 'blur_003', 'blur_1', None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.condition(value)

    def test_training_identity_requires_same_condition_same_run_and_fingerprint(self):
        c = self.module()
        run = str(uuid.uuid4())
        saved = c.training_identity('blur_004', run)
        c.validate_training_identity(saved, 'blur_004', run)
        for key, bad in [('condition_id', 'blur_008'), ('run_id', str(uuid.uuid4())),
                         ('condition_fingerprint', '0' * 64)]:
            other = copy.deepcopy(saved)
            other[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_training_identity(other, 'blur_004', run)

    def test_run_relative_path_rejects_escape_noncanonical_uuid(self):
        c = self.module()
        run = '11111111-1111-4111-8111-111111111111'
        self.assertEqual(c.run_relative(run), 'work/runs/' + run + '/training')
        for value in ('../other', run.upper().replace('1', 'A', 1), '', 'not-uuid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.run_relative(value)

    def test_fresh_command_has_new_condition_and_seed_and_no_resume(self):
        c = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = str(uuid.uuid4())
            argv = c.training_argv(root, Path('python.exe'), Path('research'), 'mosaic_000', run)
            self.assertEqual(argv[argv.index('--condition-id') + 1], 'mosaic_000')
            self.assertEqual(argv[argv.index('--seed') + 1], '3')
            self.assertEqual(argv[argv.index('--tasks') + 1], 'both')
            self.assertNotIn('--resume', argv)
            self.assertEqual(argv[argv.index('--output') + 1], str(root / c.run_relative(run)))

    def test_identity_config_cannot_claim_original_model_reuse(self):
        c = self.module()
        value = c.training_identity('mosaic_001', str(uuid.uuid4()))
        self.assertEqual(value['initialization_mode'], 'fresh_imagenet_seed3')
        self.assertIs(value['cross_condition_state_inherited'], False)
        self.assertEqual(value['condition']['operator_mode'], 'original')

    def test_atomic_state_is_strict_json_and_does_not_replace_on_nonfinite_input(self):
        c = self.module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            c.atomic_json(path, {'state': 'waiting_baseline'})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                c.atomic_json(path, {'loss': float('nan')})
            self.assertEqual(path.read_bytes(), original)

    def test_queue_state_uses_twenty_unique_runs_without_starting_work(self):
        c = self.module()
        state = c.initial_queue_state(str(uuid.uuid4()), {'run_id': str(uuid.uuid4())})
        self.assertEqual(state['status'], 'waiting_baseline')
        self.assertIsNone(state['current_condition_id'])
        self.assertEqual(len(state['conditions']), 20)
        self.assertTrue(all(row['status'] == 'pending' and row['run_id'] is None for row in state['conditions']))


class QueueGateTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('cxr10_queue'), '안전 순차 큐 구현이 필요합니다.')
        import cxr10_queue
        return cxr10_queue

    def test_live_or_failed_baseline_never_opens_gpu_gate(self):
        q = self.module()
        self.assertEqual(q.baseline_exit_gate(None), 'waiting_baseline')
        for code in (1, 130, 259, 3221226505):
            with self.subTest(code=code), self.assertRaises(ValueError):
                q.baseline_exit_gate(code)
        self.assertEqual(q.baseline_exit_gate(0), 'ready')

    def test_pid_reuse_or_other_command_is_not_same_baseline(self):
        q = self.module()
        expected = {'pid': 77, 'creation_filetime': 1234, 'command_line': 'correct command'}
        self.assertTrue(q.same_process(expected, dict(expected)))
        for key, bad in [('pid', 78), ('creation_filetime', 1235), ('command_line', 'other command')]:
            changed = dict(expected)
            changed[key] = bad
            self.assertFalse(q.same_process(expected, changed))

    def test_lock_prevents_second_queue_and_releases_after_exit(self):
        q = self.module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queue.lock'
            with q.QueueLock(path):
                with self.assertRaises((OSError, RuntimeError)):
                    with q.QueueLock(path):
                        self.fail('두 번째 큐가 lock을 얻었습니다.')
            with q.QueueLock(path):
                pass

    def test_condition_child_failure_is_not_completion_or_next_condition(self):
        q = self.module()
        for code in (1, 130, -1):
            with self.subTest(code=code), self.assertRaises(RuntimeError):
                q.require_child_success(code, 'blur_001')
        q.require_child_success(0, 'blur_001')


if __name__ == '__main__':
    unittest.main()
