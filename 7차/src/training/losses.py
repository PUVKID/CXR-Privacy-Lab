"""Checkpoint adapter around the author's pinned PML 0.9.94 loss.

The vendor kernel is unmodified. In particular, float16 distance computation
and validation enqueues follow that version, not a float32 approximation.
"""

import torch
from torch import nn
from ._vendor.pytorch_metric_learning.losses import ContrastiveLoss, CrossBatchMemory


class CrossBatchContrastive(nn.Module):
    """PML loss with recoverable buffers, preserving validation enqueues.

The author also enqueues validation embeddings in the same queue. This is
retained for source-method fidelity, not promoted as a new validation design.
"""

    def __init__(self, embedding_size=128, memory_size=128):
        super().__init__()
        if embedding_size < 1 or memory_size < 1:
            raise ValueError("Embedding and queue sizes must be positive")
        self.kernel = CrossBatchMemory(ContrastiveLoss(), embedding_size,
                                       memory_size=memory_size)
        self.register_buffer("memory", torch.zeros(memory_size, embedding_size))
        self.register_buffer("labels", torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, embeddings, labels):
        if len(embeddings) < 1 or len(embeddings) > len(self.memory):
            raise ValueError("Batch must be nonempty and no larger than CBM memory")
        self.kernel.embedding_memory = self.memory.to(embeddings.device, embeddings.dtype)
        self.kernel.label_memory = self.labels.to(labels.device, labels.dtype)
        self.kernel.queue_idx = int(self.cursor)
        self.kernel.has_been_filled = int(self.count) == len(self.memory)
        result = self.kernel(embeddings, labels)
        self.memory = self.kernel.embedding_memory
        self.labels = self.kernel.label_memory
        self.cursor.fill_(self.kernel.queue_idx)
        self.count.fill_(min(len(self.memory), int(self.count) + len(embeddings)))
        return result
