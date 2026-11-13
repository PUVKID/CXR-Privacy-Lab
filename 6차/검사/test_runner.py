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
    def scenario(self, returncodes, cancel=False, foreign=False, validated=False):
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp) / '5차'
            (here / '실험 코드').mkdir(parents=True)
            control = here / 'control'
            stopped, started = [], []
            class Child:
                def __init__(self, argv, **kwargs):
                    self.argv = argv
                    self.cwd = kwargs['cwd']
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
            with patch.multiple(runner, HERE=here, BASE=Path(tmp)/'연구 자료', CONTROL=control, PYTHON=Path('fixture-python')), patch.object(runner.subprocess,'Popen',Child), patch.object(runner,'stop_child',lambda c:stopped.append(c.pid)):
                snapshot = None
                if validated:
                    snapshot = Path(tmp)/'연구 자료/validation_runs/fixture-snapshot'
                    snapshot.mkdir(parents=True)
                code = runner.main(validated_snapshot=snapshot)
            state = json.loads((control/'run.json').read_text(encoding='utf-8'))
            return code,state,started,stopped

    def test_new_training_precedes_evaluation_and_uses_its_weights(self):
        code,state,started,stopped = self.scenario([0,0,0,0])
        self.assertEqual((code,state['status']),(0,'completed'))
        self.assertEqual(state['revision'],5)
        self.assertEqual(len(started),4)
        audit_arg=started[1].argv[started[1].argv.index('--reuse-nih-pixel-manifest')+1]
        self.assertEqual(Path(audit_arg).parts[-3:],('validation_runs','원본_전수검사','nih_pixels.private.json'))
        self.assertIn('--output',started[2].argv)
        self.assertIn('--verification-checkpoint',started[3].argv)
        self.assertIn('--validated-snapshot',started[2].argv)
        self.assertIn('--activation-checkpointing',started[2].argv)
        self.assertEqual(started[2].argv[started[2].argv.index('--num-workers')+1], '0')
        self.assertTrue(all(child.cwd.name == '실험 코드' for child in started))
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

    def test_validated_snapshot_failure_cannot_start_training(self):
        code,state,started,stopped=self.scenario([0,1],validated=True)
        self.assertEqual((code,state['status']),(1,'failed'))
        self.assertEqual(len(started),2)
        self.assertIn('validate_prepared_snapshot',' '.join(started[1].argv))
        self.assertTrue(state['validation_snapshot'].endswith('fixture-snapshot'))
        self.assertFalse(any('train_from_scratch.py' in argument for child in started for argument in child.argv))

    def test_validated_snapshot_is_rechecked_and_shared_by_training_and_evaluation(self):
        code,state,started,stopped=self.scenario([0,0,0,0],validated=True)
        self.assertEqual((code,state['status']),(0,'completed'))
        self.assertIn('validate_prepared_snapshot',' '.join(started[1].argv))
        for child in started[2:]:
            self.assertEqual(child.argv[child.argv.index('--validated-snapshot')+1],state['validation_snapshot'])
        self.assertFalse(any('download_datasets.py' in argument for child in started for argument in child.argv))

if __name__ == '__main__':
    unittest.main()
