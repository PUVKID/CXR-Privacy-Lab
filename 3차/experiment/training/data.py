"""검증 전용 분할과 공식 블록 샘플링을 보존합니다."""

from collections import Counter
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from cxr_reid.data import read_pair_manifest, read_retrieval_manifest


def patient(name):
    return str(int(Path(name).name.split("_")[0]))


def assert_patient_disjoint(*names_by_phase):
    groups = [{patient(name) for name in names} for names in names_by_phase]
    for i, group in enumerate(groups):
        for other in groups[i + 1:]:
            if group & other:
                raise ValueError("학습·검증·평가 분할 사이에 환자 중복이 있습니다.")


def check_pairs(pairs, allowed):
    allowed = set(allowed)
    for first, second, label in pairs:
        if first not in allowed or second not in allowed:
            raise ValueError("쌍에 해당 분할 밖의 영상이 있습니다.")
        if first == second or (patient(first) == patient(second)) != bool(label):
            raise ValueError("동일 파일 또는 환자 라벨이 잘못된 쌍이 있습니다.")


def randomized_pairs(positive_pairs, names, count, *, seed):
    """공식 RNP처럼 고정 양성 쌍과 에폭별 무작위 음성 쌍을 결합합니다."""
    if count <= 0 or count % 2 or len(positive_pairs) < count // 2:
        raise ValueError("균형 쌍 수는 양의 짝수이며 충분한 양성 쌍이 필요합니다.")
    if len({patient(name) for name in names}) < 2:
        raise ValueError("음성 쌍 생성에는 서로 다른 환자가 두 명 이상 필요합니다.")
    rng = random.Random(seed)
    positives = list(positive_pairs[:count // 2])
    negatives = []
    while len(negatives) < count // 2:
        first, second = rng.choice(names), rng.choice(names)
        if patient(first) != patient(second):
            negatives.append((first, second, 0))
    # 공식 코드도 생성한 음성 쌍을 random.sample로 다시 섞습니다.
    result = positives + rng.sample(negatives, len(negatives))
    check_pairs(result, names)
    return result


def mining_order(rows, *, seed, block_size=8):
    """공식 build_mining_list_32/get_block의 블록8 규칙을 이식합니다."""
    if block_size != 8:
        raise ValueError("공식 재현 샘플러는 블록 크기 8만 허용합니다.")
    names = np.arange(len(rows))
    labels = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    if any(value < 2 for value in Counter(labels).values()):
        raise ValueError("검색 학습에는 원본 train.csv의 다중 영상 환자만 허용합니다.")
    rng = np.random.RandomState(seed)
    coin = random.Random(seed)
    output, remainder = [], []

    def take(current, count):
        nonlocal names, labels
        chosen = rng.choice(names[labels == current], int(count), replace=False)
        keep = ~np.isin(names, chosen)
        names, labels = names[keep], labels[keep]
        return chosen.tolist()

    while len(names):
        current = rng.choice(labels)
        count = int(np.sum(labels == current))
        block = []
        success = True
        if count % block_size == 0:
            block += take(current, block_size)
        else:
            if count > block_size:
                if count % 8 == 1:
                    block += take(current, 2)
                elif coin.random() < .5:
                    block += take(current, block_size)
                elif count % block_size == block_size - 1:
                    block += take(current, count % block_size - 2)
                else:
                    block += take(current, count % block_size)
            else:
                block += take(current, count % block_size)
            rest = block_size - len(block)
            while rest:
                ids, counts = np.unique(labels, return_counts=True)
                mod = counts % block_size
                possible = ids[(mod > 1) & (((rest - mod) > 1) | ((rest - mod) == 0))]
                if not len(possible):
                    possible = ids[mod == block_size - 1]
                    if not len(possible):
                        success = False
                    else:
                        block += take(rng.choice(possible), rest)
                    break
                current = rng.choice(possible)
                block += take(current, int(mod[ids == current].item()))
                rest = block_size - len(block)
        (output if success else remainder).extend(block)
    result = output + remainder
    if len(result) != len(rows) or len(set(result)) != len(rows):
        raise RuntimeError("공식 블록 샘플링에서 영상이 누락되거나 중복되었습니다.")
    return result


def load_protocol(official):
    """서로 다른 verification/retrieval 분할 목록을 섞지 않습니다."""
    official = Path(official)
    vr, rr = official / "PatientVerification", official / "PatientRetrieval/csv_files"
    train_val = (vr / "train_val_list.txt").read_text(encoding="utf-8").split()
    names = {"train": train_val[:75708], "validation": train_val[75708:],
             "test": (vr / "test_list.txt").read_text(encoding="utf-8").split()}
    if [len(names[key]) for key in ("train", "validation", "test")] != [75708, 10816, 25596]:
        raise ValueError("원 저자의 검증 전용 영상 분할 수가 다릅니다.")
    assert_patient_disjoint(*names.values())
    pairs = {}
    for phase, filename, count in (("train", "pairs_training.txt", 792294),
                                   ("validation", "pairs_validation.txt", 50000),
                                   ("test", "pairs_testing.txt", 100000)):
        pairs[phase] = read_pair_manifest(vr / "image_pairs" / filename, expected_count=count)
        check_pairs(pairs[phase], names[phase])
    retrieval = {
        "train": read_retrieval_manifest([rr / "train.csv"]),
        "validation": read_retrieval_manifest([rr / "validation.csv", rr / "validation_single_files.csv"]),
        "test": read_retrieval_manifest([rr / "test.csv", rr / "test_single_files.csv"]),
    }
    assert_patient_disjoint(*[[name for name, _ in rows] for rows in retrieval.values()])
    for rows in retrieval.values():
        for name, label in rows:
            if patient(name) != str(int(label)):
                raise ValueError("검색 CSV의 환자 ID와 원본 영상 이름이 다릅니다.")
    return names, pairs, retrieval


class PairDataset(Dataset):
    def __init__(self, root, pairs):
        self.root, self.pairs = Path(root), pairs
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first, second, label = self.pairs[index]
        images = []
        for name in (first, second):
            with Image.open(self.root / name) as image:
                images.append(self.transform(image.convert("RGB")))
        return *images, torch.tensor(float(label))


class RetrievalDataset(Dataset):
    def __init__(self, root, rows, *, augment=False, seed=3):
        self.root, self.rows, self.seed = Path(root), rows, seed
        operations = [transforms.ColorJitter(brightness=.4, saturation=0, contrast=.4, hue=0)] if augment else []
        self.transform = transforms.Compose(operations + [transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, label = self.rows[index]
        with Image.open(self.root / name) as image:
            if image.size != (1024, 1024):
                raise ValueError("공식 검색 학습 원본은 1024×1024이어야 합니다.")
            # 에폭별·영상별 증강 seed를 고정하여 Windows 작업자 수와 재개에 독립시킵니다.
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(torch.Generator().manual_seed(self.seed + index).get_state())
                output = self.transform(image.convert("RGB"))
        return output, int(label)
