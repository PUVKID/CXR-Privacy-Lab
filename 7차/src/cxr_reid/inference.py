"""같은 환자의 촬영 시점이 다른 영상을 평가하는 배치 추론 도구."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .image_ops import process_image


class XrayDataset(Dataset):
    """원본 영상에 변환을 적용한 뒤 원 논문의 입력 규격으로 정규화한다."""

    def __init__(self, image_root: Path, names: list[str], mode: str, strength: int,
                 image_size: int, *, paper_resolution: int | None = None):
        self.root = Path(image_root)
        self.names = list(names)
        self.mode = mode
        self.strength = strength
        self.image_size = image_size
        self.paper_resolution = paper_resolution
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # 원 저자의 검증 모델은 256, 검색 모델은 원본 1024 픽셀을 사용한다.
        with Image.open(self.root / self.names[idx]) as file:
            image = file.convert("RGB")
        image = process_image(image, self.mode, self.strength)
        if self.paper_resolution and self.paper_resolution != 1024:
            # 원 논문의 해상도 실험은 축소 후 원래 1024로 복원한다.
            image = image.resize((self.paper_resolution,) * 2, Image.Resampling.BILINEAR)
            image = image.resize((1024, 1024), Image.Resampling.BILINEAR)
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return self.normalize(image)


@torch.inference_mode()
def extract_embeddings(model, image_root: Path, names: list[str], *,
                       task: str, mode: str, strength: int, device: str,
                       batch_size: int, num_workers: int, paper_resolution: int | None = None,
                       progress_callback=None) -> np.ndarray:
    """검증 모델과 검색 모델에서 각각 논문의 128차원 표현을 얻는다."""
    if task not in ("verification", "retrieval"):
        raise ValueError(f"알 수 없는 모델 유형: {task}")
    if not names:
        raise ValueError("추론할 영상 names가 비어 있습니다 (empty input)")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size>=1, num_workers>=0 required")
    data = XrayDataset(image_root, names, mode, strength,
                       256 if task == "verification" else 1024,
                       paper_resolution=paper_resolution)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.startswith("cuda"),
                        persistent_workers=num_workers > 0)
    outputs = []
    if progress_callback is not None:
        progress_callback(0, len(loader))
    for batch_index, images in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if task == "verification":
            embedding = model.encode(images)
        else:
            embedding = model(images)
        outputs.append(embedding.detach().cpu().numpy())
        if progress_callback is not None and (batch_index == 0 or (batch_index + 1) % 25 == 0 or batch_index + 1 == len(loader)):
            progress_callback(batch_index + 1, len(loader))
        if batch_index % 50 == 0 or batch_index + 1 == len(loader):
            print(f"    {task} {mode} {strength}: {batch_index+1:,}/{len(loader):,} 배치", flush=True)
    return np.concatenate(outputs).astype(np.float32)


def predict_pair_probabilities(model, pairs, file_to_index: dict, query_features, original_features):
    """원본 시험 쌍을 변경하지 않고 비대칭(query 변형, gallery 원본)으로 비교한다."""
    image_1 = np.fromiter((file_to_index[p[0]] for p in pairs), dtype=np.int32, count=len(pairs))
    image_2 = np.fromiter((file_to_index[p[1]] for p in pairs), dtype=np.int32, count=len(pairs))
    # 공식 네트워크의 마지막 완전연결층을 동일한 계산식으로 적용한다.
    final_weights = model.fc_end.weight.detach().cpu().numpy().reshape(-1)
    final_bias = float(model.fc_end.bias.detach().cpu().numpy().item())
    raw = (np.abs(query_features[image_1] - original_features[image_2]) @ final_weights) + final_bias
    raw = np.clip(raw, -80, 80)
    probability = 1 / (1 + np.exp(-raw))
    return np.array([p[2] for p in pairs], dtype=np.int8), probability
