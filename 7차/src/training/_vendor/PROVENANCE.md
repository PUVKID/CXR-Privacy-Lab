# pytorch-metric-learning 0.9.94

Repository: https://github.com/KevinMusgrave/pytorch-metric-learning

Tag: `v0.9.94`

Commit: `f04e38c496c5c271381bb559795cd32ae4f68744`

The `pytorch_metric_learning` directory is copied byte-for-byte from the
upstream `src/pytorch_metric_learning` directory. Its MIT license is included
as `PML-LICENSE.txt`. The parent research code is licensed separately.

The original CrossBatchMemory does not register its queue as checkpoint
buffers. `training/losses.py` adds a serializable buffer adapter around this
unmodified upstream implementation. Loss, distance, reduction, and memory
enqueue behavior are delegated to upstream, including enqueue during
validation. No installed recent pytorch-metric-learning package is substituted.

Python 3.12 may emit harmless invalid-regex-escape SyntaxWarnings from the
historical code; PyTorch 2.x may emit a meshgrid deprecation warning. These
are upstream compatibility warnings, not an assertion of identical GPU bits.
