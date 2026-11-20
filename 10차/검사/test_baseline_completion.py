"""기존7차 학습·외부평가·84공격평가의 전체 완료 근거를 합성 파일로 검사합니다."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cxr10_contract as c
import cxr10_queue as q
from cxr7_contract import EXPECTED_SHA256
from cxr7_privacy_contract import condition_plan
from cxr7_reports import evaluator_fingerprint


RUN = '22222222-2222-4222-8222-222222222222'


class BaselineCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '10차'
        self.root.mkdir()
        self.seven = self.root.parent / '7차'
        self.output = self.seven / 'runs' / RUN
        self.output.mkdir(parents=True)
        source_names = {'cxr7_evaluate.py', 'cxr7_reports.py', 'cxr7_privacy.py', 'cxr7_privacy_contract.py',
            'cxr7_examples.py', 'cxr_reid/models.py', 'cxr_reid/evaluation.py', 'cxr_reid/inference.py',
            'cxr_reid/image_ops.py', 'cxr_reid/data.py'}
        for name in source_names:
            path = self.seven / 'src' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# 합성 코드 표지\n', encoding='utf-8')
        artifacts = {'published_cxr_initialization': False}
        hashes = {}
        for task, phase in (('verification', 'verification'), ('retrieval', 'retrieval_phase2')):
            path = self.output / phase / 'selected_weights.pth'
            path.parent.mkdir()
            path.write_bytes(task.encode('ascii'))
            hashes[task] = hashlib.sha256(task.encode('ascii')).hexdigest()
            artifacts[task + '_sha256'] = hashes[task]
        c.atomic_json(self.output / 'artifacts.json', artifacts)
        self.verification = {'auc': .8, 'accuracy': .7, 'confidence_interval': {'bootstrap_repeats': 10000}}
        self.retrieval = {'precision_at_1': .5, 'map_at_r': .4, 'r_precision': .3}
        evaluation = {'completed': True, 'evaluator_code_sha256': evaluator_fingerprint(self.seven / 'src')}
        for dataset in ('NIH', 'COVID19'):
            evaluation[dataset] = {'verification': copy.deepcopy(self.verification),
                'retrieval': copy.deepcopy(self.retrieval), 'verification_sha256': hashes['verification'],
                'retrieval_sha256': hashes['retrieval']}
        c.atomic_json(self.output / 'evaluation_results.json', evaluation)
        privacy_sources = source_names - {'cxr7_evaluate.py', 'cxr7_reports.py'}
        payload = {'checkpoints_sha256': hashes, 'baseline_results_sha256': c.file_sha256(self.output / 'evaluation_results.json'),
            'official_manifests': EXPECTED_SHA256, 'plan': condition_plan(),
            'sources': {name: c.file_sha256(self.seven / 'src' / name) for name in privacy_sources}}
        privacy = {'completed': True, 'binding_payload': payload, 'results': {}}
        for dataset in ('NIH', 'COVID19'):
            privacy['results'][dataset] = {row['id']: {'verification_status': 'completed',
                'retrieval_status': 'completed', 'verification': copy.deepcopy(self.verification),
                'retrieval': copy.deepcopy(self.retrieval)} for row in condition_plan()}
        privacy['binding'] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
            allow_nan=False).encode('utf-8')).hexdigest()
        c.atomic_json(self.output / 'privacy_results.json', privacy)

    def change(self, name, update):
        path = self.output / name
        value = c.read_json(path)
        update(value)
        c.atomic_json(path, value)

    def test_full_original_outputs_are_accepted_without_reading_medical_images(self):
        result = q.baseline_artifact_gate(self.root, RUN)
        self.assertEqual(result['completed_task_conditions'], 84)

    def test_training_artifacts_alone_do_not_open_gate(self):
        (self.output / 'evaluation_results.json').unlink()
        with self.assertRaises(FileNotFoundError):
            q.baseline_artifact_gate(self.root, RUN)

    def test_original_evaluation_incomplete_blocks(self):
        self.change('evaluation_results.json', lambda value: value.update(completed=False))
        with self.assertRaises(ValueError):
            q.baseline_artifact_gate(self.root, RUN)

    def test_one_missing_attack_of84_blocks_even_when_summary_claims_completed(self):
        self.change('privacy_results.json', lambda value: value['results']['NIH']['blur_004'].update(retrieval_status='pending'))
        with self.assertRaises(ValueError):
            q.baseline_artifact_gate(self.root, RUN)

    def test_bootstrap_not_finished_blocks(self):
        self.change('evaluation_results.json', lambda value: value['NIH']['verification'].pop('confidence_interval'))
        with self.assertRaises(ValueError):
            q.baseline_artifact_gate(self.root, RUN)

    def test_changed_original_model_blocks(self):
        (self.output / 'verification/selected_weights.pth').write_bytes(b'other')
        with self.assertRaises(ValueError):
            q.baseline_artifact_gate(self.root, RUN)

    def test_changed_original_evaluation_code_blocks(self):
        (self.seven / 'src/cxr7_privacy.py').write_text('# 코드 변경\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            q.baseline_artifact_gate(self.root, RUN)


if __name__ == '__main__':
    unittest.main()
