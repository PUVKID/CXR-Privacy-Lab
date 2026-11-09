"""실데이터를 건드리지 않고 실행 관리자 실패·중단 경계를 검사합니다."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'run_experiment.py'
spec = importlib.util.spec_from_file_location('cxr_runner', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerContract(unittest.TestCase):
    def scenario(self, returncodes, cancel=False):
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp) / '1차'
            (here / 'experiment').mkdir(parents=True)
            (here / 'experiment/requirements.txt').write_text('fixture', encoding='utf-8')
            control = Path(tmp) / 'research/control'
            stopped = []
            started = []
            class Child:
                def __init__(self, argv, **kwargs):
                    self.argv = argv
                    self.pid = 9000 + len(started)
                    self.returncode = returncodes[len(started)]
                    started.append(self)
                    if cancel:
                        state = json.loads((control / 'run.json').read_text(encoding='utf-8'))
                        (control / 'stop.request').write_text(state['run_id'], encoding='utf-8')
                def poll(self):
                    return self.returncode
            with patch.multiple(runner, HERE=here, BASE=control.parent, CONTROL=control,
                                PYTHON=Path('fixture-python')), \
                 patch.object(runner.subprocess, 'Popen', Child), \
                 patch.object(runner, 'stop_child', lambda child: stopped.append(child.pid)):
                result = runner.main()
            state = json.loads((control / 'run.json').read_text(encoding='utf-8'))
            return result, state, started, stopped, (control / 'environment_ready.json').exists()

    def test_success_requires_all_processes(self):
        code, state, started, stopped, marker = self.scenario([0, 0, 0, 0, 0])
        self.assertEqual((code, state['status']), (0, 'completed'))
        self.assertEqual(len(started), 5)
        self.assertIn('--skip-download', started[-1].argv)
        self.assertTrue(marker)
        self.assertFalse(stopped)

    def test_environment_failure_cannot_start_evaluation(self):
        code, state, started, stopped, marker = self.scenario([None, 1])
        self.assertEqual((code, state['status']), (1, 'failed'))
        self.assertEqual(len(started), 2)
        self.assertEqual(set(stopped), {9000, 9001})
        self.assertFalse(marker)

    def test_dataset_failure_cannot_become_completion(self):
        code, state, started, stopped, marker = self.scenario([1, 0, 0, 0])
        self.assertEqual((code, state['status']), (1, 'failed'))
        self.assertEqual(len(started), 4)
        self.assertIn('NIH', state['error'])

    def test_cancel_stops_owned_children_and_retains_state(self):
        code, state, started, stopped, marker = self.scenario([None, None], cancel=True)
        self.assertEqual((code, state['status']), (130, 'stopped'))
        self.assertEqual(set(stopped), {9000, 9001})
        self.assertIsNotNone(state['finished_at'])
        self.assertFalse(marker)


if __name__ == '__main__':
    unittest.main()
