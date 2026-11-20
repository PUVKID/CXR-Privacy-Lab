"""사용자 정정으로 추가한 원본 새 시작 분기만 최소 검사합니다."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cxr10_contract as c
import cxr10_queue as q


class FreshRestartTests(unittest.TestCase):
    def test_original_is_first_fresh_condition_separate_from_identity_strengths(self):
        rows = c.condition_plan()
        self.assertEqual(len(rows), 21)
        self.assertEqual(rows[0]['id'], 'original_000')
        self.assertEqual(rows[0]['operator_mode'], 'original')
        self.assertEqual(len({c.condition_fingerprint(r['id']) for r in rows}), 21)
        argv = c.training_argv(Path('root'), Path('python.exe'), Path('data'),
            'original_000', '11111111-1111-4111-8111-111111111111')
        self.assertNotIn('--resume', argv)
        self.assertNotIn('--no-wait-for-memory', argv)
        self.assertEqual(argv[argv.index('--seed') + 1], '3')

    def test_prepare_preserves_old_queue_and_clears_stop_without_old_handle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / 'work/queue'
            old.mkdir(parents=True)
            c.atomic_json(old / 'state.json', {'conditions': []})
            (old / 'plan.private.json').write_bytes(b'{"old":true}')
            (old / 'STOP_REQUESTED').write_bytes(b'old stop')
            with patch.object(q, 'require_no_cxr_processes'), patch.object(q, 'ProcessHandle', side_effect=AssertionError('이전 핸들 금지')):
                plan = q.prepare_fresh(root)
            self.assertEqual(plan['start_mode'], 'fresh_all_21')
            archive = root / plan['previous_queue_archive']
            self.assertEqual((archive / 'plan.private.json').read_bytes(), b'{"old":true}')
            self.assertEqual((archive / 'STOP_REQUESTED').read_bytes(), b'old stop')
            self.assertFalse((old / 'STOP_REQUESTED').exists())
            state = c.read_json(old / 'state.json')
            self.assertEqual(state['status'], 'ready')
            self.assertEqual(state['conditions'][0]['id'], 'original_000')
            self.assertTrue(all(r['run_id'] is None for r in state['conditions']))

    def test_fresh_execution_skips_baseline_and_launches_original_first(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(q, 'require_no_cxr_processes'):
                plan = q.prepare_fresh(root)
            manager = SimpleNamespace(identity={'pid': 1})
            launches = []
            def stop_after_launch(root, state, row, argv, stage):
                launches.append((row['id'], stage))
                raise RuntimeError('합성 첫 자식 진입 확인')
            with patch.object(q, 'prevent_sleep', return_value=nullcontext()), \
                 patch.object(q, 'ProcessHandle', return_value=nullcontext(manager)), \
                 patch.object(q, 'wait_baseline', side_effect=AssertionError('7차 대기 금지')), \
                 patch.object(q, 'baseline_artifact_gate', side_effect=AssertionError('7차 평가 의존 금지')), \
                 patch.object(q, 'require_no_cxr_processes'), patch.object(q, 'run_child', side_effect=stop_after_launch):
                with self.assertRaisesRegex(RuntimeError, '합성 첫 자식'):
                    q.execute(root, SimpleNamespace(resume_condition=None, poll_seconds=5))
            self.assertEqual(launches, [('original_000', 'training')])

    def test_wddm_gui_or_other_project_gpu_context_is_observation_not_cxr_block(self):
        state = {}
        with patch.object(q, 'cxr_processes', return_value=[]), patch.object(q, 'gpu_processes', return_value=[12, 34]):
            q.require_no_cxr_processes(Path('root'), state)
        self.assertEqual(state['gpu_observation']['pids'], [12, 34])
        with patch.object(q, 'cxr_processes', return_value=[{'pid': 99}]), patch.object(q, 'gpu_processes', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'CXR'):
                q.require_no_cxr_processes(Path('root'))

    def test_cim_filter_distinguishes_cxr_training_from_gui_and_other_project(self):
        root = Path('E:/2026/흉부 엑스레이 환자 재식별 실험/10차')
        rows = [{'ProcessId': 1, 'ParentProcessId': 0, 'CommandLine': 'chrome.exe'},
            {'ProcessId': 2, 'ParentProcessId': 0, 'CommandLine': 'python D:/other/train_from_scratch.py'},
            {'ProcessId': 3, 'ParentProcessId': 0,
             'CommandLine': 'python "' + str(root / 'src/train_from_scratch.py') + '"'}]
        result = SimpleNamespace(returncode=0, stdout=json.dumps(rows).encode('utf-8'))
        with patch.object(q.subprocess, 'run', return_value=result):
            self.assertEqual([r['pid'] for r in q.cxr_processes(root)], [3])


if __name__ == '__main__':
    unittest.main()
