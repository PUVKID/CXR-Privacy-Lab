"""새 학습이 성공해야 현재 실행의 가중치로 평가하는지 격리 검사합니다."""
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
    def scenario(self, returncodes, cancel=False, foreign=False):
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp) / '3차'
            (here / 'experiment').mkdir(parents=True)
            control = here / 'control'
            stopped, started = [], []
            class Child:
                def __init__(self, argv, **kwargs):
                    self.argv = argv
                    self.pid = 9000 + len(started)
                    self.returncode = returncodes[len(started)]
                    started.append(self)
                    if any(str(x).endswith('download_datasets.py') for x in argv) and self.returncode == 0:
                        research = Path(argv[argv.index('--project') + 1])
                        report = research / 'validation_runs' / 'fixture-snapshot' / 'report.json'
                        report.parent.mkdir(parents=True)
                        report.write_text('{}',encoding='utf-8')
                    if any(str(x).endswith('train_from_scratch.py') for x in argv) and self.returncode == 0:
                        output = Path(argv[argv.index('--output') + 1])
                        output.mkdir(parents=True)
                        items = {}
                        for key in ['verification_checkpoint', 'retrieval_checkpoint']:
                            dest = (here / 'foreign.pth') if foreign else output / (key + '.pth')
                            dest.write_bytes(b'fixture')
                            items[key] = str(dest)
                        (output / 'artifacts.json').write_text(json.dumps(items), encoding='utf-8')
                    if cancel:
                        state = json.loads((control / 'run.json').read_text(encoding='utf-8'))
                        (control / 'stop.request').write_text(state['run_id'], encoding='utf-8')
                def poll(self):
                    return self.returncode
            with patch.multiple(runner, HERE=here, BASE=Path(tmp)/'research', CONTROL=control, PYTHON=Path('fixture-python')), patch.object(runner.subprocess,'Popen',Child), patch.object(runner,'stop_child',lambda c:stopped.append(c.pid)):
                code = runner.main()
            state = json.loads((control/'run.json').read_text(encoding='utf-8'))
            return code,state,started,stopped

    def test_new_training_precedes_evaluation_and_uses_its_weights(self):
        code,state,started,stopped = self.scenario([0,0,0,0])
        self.assertEqual((code,state['status']),(0,'completed'))
        self.assertEqual(len(started),4)
        self.assertIn('--output',started[2].argv)
        self.assertIn('--verification-checkpoint',started[3].argv)
        self.assertIn('--validated-snapshot',started[2].argv)
        self.assertIn('--activation-checkpointing',started[2].argv)
        self.assertIn(state['run_id'],' '.join(started[3].argv))
        self.assertFalse(any('--resume' in arg for child in started for arg in child.argv))
        self.assertFalse(stopped)

    def test_training_failure_cannot_start_evaluation(self):
        code,state,started,stopped=self.scenario([0,0,1])
        self.assertEqual((code,state['status']),(1,'failed'))
        self.assertEqual(len(started),3)
        self.assertIn('학습',state['error'])

    def test_dataset_failure_cannot_start_training(self):
        code,state,started,stopped=self.scenario([0,1])
        self.assertEqual((code,state['status']),(1,'failed'))
        self.assertEqual(len(started),2)

    def test_foreign_checkpoint_is_rejected(self):
        code,state,started,stopped=self.scenario([0,0,0],foreign=True)
        self.assertEqual((code,state['status']),(1,'failed'))
        self.assertEqual(len(started),3)
        self.assertIn('현재 실행',state['error'])

    def test_cancel_targets_only_owned_processes(self):
        code,state,started,stopped=self.scenario([None],cancel=True)
        self.assertEqual((code,state['status']),(130,'stopped'))
        self.assertEqual(stopped,[9000])
        self.assertIsNotNone(state['finished_at'])

if __name__ == '__main__':
    unittest.main()
