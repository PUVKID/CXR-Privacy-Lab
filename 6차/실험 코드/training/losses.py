"""PML 0.9.94의 정규화된 비제곱 contrastive/CBM128을 이식합니다."""

import torch
from torch import nn
from torch.nn import functional as F


def contrastive_loss(embeddings, labels, reference=None, reference_labels=None, self_columns=None):
    reference = embeddings if reference is None else reference
    reference_labels = labels if reference_labels is None else reference_labels
    left = F.normalize(embeddings.float(), p=2, dim=1)
    right = F.normalize(reference.float(), p=2, dim=1)
    distance = torch.cdist(left, right, p=2)
    positive = labels[:, None] == reference_labels[None, :]
    negative = ~positive
    if self_columns is None and reference is embeddings:
        positive.fill_diagonal_(False)
    elif self_columns is not None:
        positive[torch.arange(len(labels), device=labels.device), self_columns] = False
    losses = (distance[positive], F.relu(1. - distance[negative]))
    result = embeddings.sum() * 0.
    for values in losses:
        nonzero = values[values > 0]
        if nonzero.numel():
            result = result + nonzero.mean()
    return result


class CrossBatchContrastive(nn.Module):
    def __init__(self, embedding_size=128, memory_size=128):
        super().__init__()
        self.register_buffer("memory", torch.zeros(memory_size, embedding_size))
        self.register_buffer("labels", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, embeddings, labels):
        if not self.training:
            # 검증 영상이 다음 학습 배치의 메모리로 유입되는 원본 부작용을 차단합니다.
            return contrastive_loss(embeddings, labels)
        size = len(embeddings)
        if size > len(self.memory):
            raise ValueError("배치 크기가 cross-batch memory보다 큽니다.")
        indices = (torch.arange(size, device=embeddings.device) + self.cursor) % len(self.memory)
        with torch.no_grad():
            self.memory[indices] = embeddings.detach().to(self.memory.dtype)
            self.labels[indices] = labels
            self.cursor.copy_((self.cursor + size) % len(self.memory))
            self.count.copy_(torch.minimum(self.count + size, self.count.new_tensor(len(self.memory))))
        count = int(self.count)
        return contrastive_loss(embeddings, labels, self.memory[:count].clone(),
                                self.labels[:count], self_columns=indices)
