"""현대 PyTorch에서 원 논문 학습 조건을 명시적으로 실행합니다."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50

from cxr_reid.models import VerificationNetwork, ExtractorRes50, RetrievalHead
from cxr_reid.evaluation import calculate_retrieval_metrics
from cxr7_contract import (assert_training_scope, audit_official_manifests,
                           audit_image_presence)
from .data import PairDataset, RetrievalDataset, load_protocol, randomized_pairs, mining_order
from .losses import CrossBatchContrastive
from .imagenet import ensure_safe_initialization
from .state import atomic_json, atomic_torch_save, digest, load_checkpoint, progress, save_checkpoint, rng_state, restore_rng
from cxr10_contract import training_identity, protocol_fingerprint, validate_training_identity


IMAGENET_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
PROTOCOL = {
    "source_commit": "880168f564afa07936a8d8fb3b868b140ab937d9",
    "initialization": IMAGENET_URL, "published_cxr_initialization": False,
    "batch_size": 32, "verification_resolution": 256, "retrieval_resolution": 1024,
    "verification_pairs_official": 792294, "verification_validation_pairs_official": 50000,
    "dataset_cleaning_policy": "UNMODIFIED_AUTHOR_SPLITS",
    "rnp_pair_count": 792294, "verification_data_handling": "randomized",
    "verification_validation_regenerated_each_epoch": True,
    "verification_epochs": 100, "verification_patience": 5, "verification_lr": .0001,
    "retrieval_phase1_epochs": 30, "retrieval_phase2_epochs": 50,
    "retrieval_lr": .158489, "weight_decay": 1e-5, "memory_size": 128,
    "mining_block_size": 8, "retrieval_precision": "float16-mixed",
    "verification_workers": 16, "retrieval_workers": 4,
    "pytorch_metric_learning": "0.9.94",
    "pytorch_metric_learning_commit": "f04e38c496c5c271381bb559795cd32ae4f68744",
    "retrieval_validation_batch_size": 32,
    "retrieval_validation_enqueues_memory": True,
    "retrieval_sanity_validation_batches": 2,
    "amp_overflow_policy": "native_scaler_skip_no_replay",
    "sampler_rng_policy": "epoch_seeded_author_block_rules",
    "augmentation_rng_policy": "worker_rng_color_jitter_per_visit",
    "resume_rng_limit": "mid_epoch_worker_augmentation_stream_not_bitwise_identical",
    "verification_cudnn_benchmark": True,
    "retrieval_cudnn_deterministic": True,
    "allow_tf32": False,
}


def seed_all(seed, *, phase="retrieval"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Verification explicitly enabled benchmark; the retrieval Lightning
    # trainer requested deterministic execution. V100 has no TF32, so disable
    # that additional RTX4090 precision path when running the modern port.
    is_verification = phase == "verification"
    torch.backends.cudnn.benchmark = is_verification
    torch.backends.cudnn.deterministic = not is_verification
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def preserve_bn_buffers(module):
    """체크포인트 재계산이 BN 이동통계를 두 번 갱신하지 않도록 복구합니다."""
    buffers = [(buffer, buffer.detach().clone()) for name, buffer in module.named_buffers()
               if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)


class TrainingExtractor(ExtractorRes50):
    def __init__(self, activation_checkpointing=False, backbone=None):
        # A discarded second backbone changes the random head initialization.
        nn.Module.__init__(self)
        backbone = resnet50(weights=None) if backbone is None else backbone
        self.extractor_net = nn.Sequential(*list(backbone.children())[:-2])
        self.activation_checkpointing = activation_checkpointing

    def forward(self, inputs):
        if not self.activation_checkpointing or not self.training or not any(p.requires_grad for p in self.parameters()):
            return self.extractor_net(inputs)
        layers = list(self.extractor_net.children())
        blocks = [nn.Sequential(*layers[:4])]
        blocks.extend(block for layer in layers[4:] for block in layer)
        for block in blocks:
            # 비재진입 방식은 입력 영상에 requires_grad를 붙이지 않아도 매개변수 기울기를 보존합니다.
            inputs = checkpoint(block, inputs, use_reentrant=False, preserve_rng_state=True,
                                context_fn=lambda block=block: (nullcontext(), preserve_bn_buffers(block)))
        return inputs


def create_model(task, *, imagenet_state=None, activation_checkpointing=False):
    if task == "verification":
        model = VerificationNetwork()
        if imagenet_state is not None:
            state = {key: value for key, value in imagenet_state.items()
                     if not key.startswith("fc.")}
            incompatibility = model.model.load_state_dict(state, strict=False)
            if set(incompatibility.missing_keys) != {"fc.weight", "fc.bias"} or incompatibility.unexpected_keys:
                raise ValueError("ImageNet backbone state does not match ResNet-50")
        return model
    if task != "retrieval":
        raise ValueError("지원하지 않는 학습 과제입니다.")
    backbone = resnet50(weights=None)
    if imagenet_state is not None:
        backbone.load_state_dict(imagenet_state, strict=True)
    extractor = TrainingExtractor(activation_checkpointing, backbone=backbone)
    return nn.Sequential(extractor, RetrievalHead())


def imagenet_initialization(cache):
    # 학습 본체는 변환된 안전 state_dict만 weights_only=True로 읽습니다.
    return ensure_safe_initialization(cache)


def parameters_digest(model, *, head_only=False):
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        if head_only and not (name.startswith("1.") or name.startswith("fc_end") or name.startswith("model.fc")):
            continue
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def optimizer_batch(model, optimizer, scaler, forward, *, memory=None, on_overflow=None):
    """One original native-AMP batch; overflow skips its optimizer update.

The previous port replayed an overflowing batch and rolled back the CBM/BN.
Native GradScaler used by the author's Lightning trainer skips the optimizer
step, lowers the scale, and proceeds. Keep that behavior and expose it in logs.
"""
    start_scale = scaler.get_scale()
    optimizer.zero_grad(set_to_none=True)
    loss = forward()
    if not torch.isfinite(loss):
        raise FloatingPointError("학습 손실이 NaN/무한대입니다. 계속 학습하지 않습니다.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    checks = [torch.isfinite(value).all() for _, value in gradients]
    finite = bool(torch.stack(checks).all()) if checks else False
    first_bad = next((name for (name, _), good in zip(gradients, checks) if not good), None)
    if not gradients or (not finite and not scaler.is_enabled()):
        raise FloatingPointError(f"유한 기울기를 얻지 못했습니다. 최초 위치: {first_bad}")
    scaler.step(optimizer)
    scaler.update()
    result = {"gradients_finite": finite, "amp_overflows": int(not finite),
              "optimizer_applied": finite,
              "scale_before": start_scale, "scale_after": scaler.get_scale()}
    if not finite:
        if not math.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0:
            raise FloatingPointError("AMP scale became invalid")
        if on_overflow is not None:
            on_overflow({**result, "overflow_attempt": 1,
                         "first_nonfinite_parameter": first_bad})
    return loss.detach(), result


def memory_gate(args, phase):
    if args.device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("학습용 CUDA GPU가 없습니다. CPU로 자동 대체하지 않습니다.")
    gc.collect()
    torch.cuda.empty_cache()
    minimum = 9. if phase == "verification" else 19.
    started = time.monotonic()
    while True:
        free, total = torch.cuda.mem_get_info()
        if total / 2 ** 30 < minimum:
            raise RuntimeError(f"장치 전체 VRAM {total / 2**30:.2f} GiB가 사전 기준 {minimum:.1f} GiB보다 작습니다.")
        if free / 2 ** 30 >= minimum:
            return
        progress(args.output, status="waiting_for_gpu", phase=phase, epoch=0, epochs=0, step=0,
                 loss=None, free_gpu_gib=free / 2 ** 30, total_gpu_gib=total / 2 ** 30,
                 minimum_free_gpu_gib=minimum)
        if not args.wait_for_memory or time.monotonic() - started >= args.memory_wait_seconds:
            raise RuntimeError("정식 배치32를 시도할 GPU 여유 메모리가 부족합니다. 다른 작업은 종료하지 않습니다.")
        if (Path(args.output) / "STOP_REQUESTED").exists():
            raise KeyboardInterrupt("사용자가 GPU 대기를 중단했습니다.")
        time.sleep(20)


def binding(args, official):
    manifests = sorted((official / "PatientVerification").glob("*.txt"))
    manifests += sorted((official / "PatientVerification/image_pairs").glob("*.txt"))
    manifests += sorted((official / "PatientRetrieval/csv_files").glob("*.csv"))
    training_root = Path(__file__).parent
    sources = {f"training/{p.relative_to(training_root).as_posix()}": digest(p)
               for p in sorted(training_root.rglob("*.py"))}
    experiment = Path(__file__).parent.parent
    for name in ("models.py", "evaluation.py", "inference.py", "image_ops.py", "data.py", "download.py"):
        sources[f"cxr_reid/{name}"] = digest(experiment / "cxr_reid" / name)
    sources["train_from_scratch.py"] = digest(experiment / "train_from_scratch.py")
    sources["cxr7_contract.py"] = digest(experiment / "cxr7_contract.py")
    for name in ('cxr10_contract.py', 'cxr10_evaluate.py', 'cxr7_reports.py'):
        sources[name] = digest(experiment / name)
    value = {"protocol": PROTOCOL, "seed": args.seed,
             "verification_workers": args.verification_workers,
             "retrieval_workers": args.retrieval_workers,
             "activation_checkpointing": args.activation_checkpointing,
             "manifests": {str(p.relative_to(official)): digest(p) for p in manifests},
             "sources": sources,
             "environment": {package: version(package) for package in
                             ("torch", "torchvision", "Pillow", "numpy", "scikit-learn")},
             "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    value['independent_training'] = training_identity(args.condition_id, args.output.parent.name)
    value['tasks'], value['device'] = args.tasks, args.device
    return protocol_fingerprint(args.condition_id, args.output.parent.name, value), value


def loader(dataset, args, *, order=None, offset=0, seed=0):
    indices = list(range(len(dataset))) if order is None else order
    indices = indices[offset * 32:]
    workers = args.verification_workers if isinstance(dataset, PairDataset) else args.retrieval_workers
    return DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False,
                      num_workers=workers, pin_memory=args.device == "cuda",
                      generator=torch.Generator().manual_seed(seed), persistent_workers=False)


@torch.no_grad()
def validation(model, phase, protocol_data, args, epoch, *, memory=None, sanity_batches=None):
    # The CBM is persistent training state. inference_mode would create an
    # inference-only queue when the first autocast pass converts it to float16,
    # then the next training enqueue would fail. The author used no_grad.
    names, pairs, retrieval = protocol_data
    images = args.project / "data/NIH_ChestXray14/images"
    model.eval()
    if phase == "verification":
        positives = [pair for pair in pairs["validation"] if pair[2] == 1]
        current = randomized_pairs(positives, names["validation"], 2*len(positives),
                                   seed=args.seed + 1000000 + epoch)
        total, count = 0., 0
        for left, right, target in loader(PairDataset(images, current, condition_id=args.condition_id), args, seed=epoch):
            pred = model(left.to(args.device), right.to(args.device)).flatten()
            loss = F.binary_cross_entropy_with_logits(pred, target.to(args.device))
            total += float(loss)
            count += 1
        return {"validation_loss": total / count}
    rows = retrieval["validation"]
    if memory is None:
        raise ValueError("Original retrieval validation requires the training CBM")
    outputs, labels = [], []
    val_loader = DataLoader(RetrievalDataset(images, rows, condition_id=args.condition_id), batch_size=32,
                            num_workers=args.retrieval_workers, shuffle=False)
    total_loss, batch_count = 0., 0
    memory.eval()
    for batch_index, (image, label) in enumerate(val_loader):
        if sanity_batches is not None and batch_index >= sanity_batches:
            break
        with torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
            vectors = model(image.to(args.device))
            current_loss = memory(vectors, label.to(args.device))
        if not torch.isfinite(current_loss):
            raise FloatingPointError("Retrieval validation loss is not finite")
        total_loss += float(current_loss)
        batch_count += 1
    # The author computes a second FP32 embedding pass through BaseTester;
    # validation_step above alone is what updates the shared memory queue.
    for image, label in val_loader:
        outputs.append(model(image.to(args.device)).float().cpu().numpy())
        labels.extend(label.tolist())
    vectors = np.concatenate(outputs)
    result = calculate_retrieval_metrics(vectors, vectors, labels, labels, same_source=True,
                                         gallery_keys=[name for name, _ in rows])
    return {"validation_loss": total_loss / max(batch_count, 1),
            "validation_map_at_r": result["map_at_r"],
            "validation_precision_at_1": result["precision_at_1"]}


def phase_train(args, phase, protocol_data, state, image_init, run_binding, previous_weights=None):
    names, pairs, retrieval = protocol_data
    seed_all(args.seed, phase=phase)
    memory_gate(args, phase)
    is_verification = phase == "verification"
    model = create_model("verification" if is_verification else "retrieval",
                         imagenet_state=image_init if previous_weights is None else None,
                         activation_checkpointing=args.activation_checkpointing)
    if previous_weights is not None:
        if phase != 'retrieval_phase2' or Path(previous_weights).resolve() != (args.output / 'retrieval_phase1/selected_weights.pth').resolve():
            raise ValueError('검색2단계는 자기 조건·실행의1단계 최종 가중치만 이어받습니다.')
        model.load_state_dict(torch.load(previous_weights, map_location="cpu", weights_only=True), strict=True)
    if phase == "retrieval_phase1":
        for parameter in model[0].parameters():
            parameter.requires_grad_(False)
    model.to(args.device)
    if is_verification:
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        epochs, scheduler, memory = 100, None, None
        full_batches = math.ceil(2*sum(label==1 for _,_,label in pairs['train']) / 32)
    else:
        groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(t in n for t in ("bias", "bn"))], "weight_decay": 1e-5},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(t in n for t in ("bias", "bn"))], "weight_decay": 0.}]
        optimizer = torch.optim.SGD(groups, lr=.158489, momentum=.9, weight_decay=1e-5)
        epochs = 30 if phase == "retrieval_phase1" else 50
        full_batches = math.ceil(len(retrieval['train']) / 32)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.158489,
            total_steps=full_batches * epochs + 1, pct_start=.3, anneal_strategy="cos",
            cycle_momentum=True, base_momentum=.85, max_momentum=.95, div_factor=25., final_div_factor=10000.)
        memory = CrossBatchContrastive().to(args.device)
    scaler = torch.amp.GradScaler("cuda", enabled=(not is_verification and args.device == "cuda"))
    folder = args.output / phase
    folder.mkdir(parents=True, exist_ok=True)
    last, exported = folder / "last.pt", folder / "selected_weights.pth"
    identity = training_identity(args.condition_id, args.output.parent.name)
    metadata = {**identity, "binding": run_binding, "phase": phase, "epoch": 0, "batch": 0,
                "global_step": 0, "train_sum": 0., "train_samples": 0,
                "best_loss": None, "bad_epochs": 0, "finished": False}
    if args.resume and last.exists():
        metadata = load_checkpoint(last, model, optimizer, binding=run_binding,
                                   scheduler=scheduler, scaler=scaler, memory=memory)
        validate_training_identity(metadata, args.condition_id, args.output.parent.name)
        if metadata.get('phase') != phase:
            raise ValueError('같은 조건이라도 다른 학습 단계의 체크포인트를 재개할 수 없습니다.')
    else:
        atomic_json(folder / "initialization.json", {**identity, "protocol_fingerprint": run_binding,
            "phase": phase, "initialization": state,
            "global_step": 0, "optimizer_state_entries": len(optimizer.state),
            "head_sha256": parameters_digest(model, head_only=True),
            "previous_phase_weights": str(previous_weights) if previous_weights else None})
    if metadata["finished"]:
        return exported
    if not is_verification and not metadata.get("sanity_completed", False):
        # Lightning 1.0.5's default num_sanity_val_steps=2 is observable because
        # the author's validation_step mutates the same CBM as training.
        progress(args.output, phase=phase, status="sanity_validation", epoch=0,
                 epochs=epochs, step=0, loss=None)
        validation(model, phase, protocol_data, args, 0, memory=memory, sanity_batches=2)
        metadata["sanity_completed"] = True
        save_checkpoint(last, model, optimizer, metadata=metadata, scheduler=scheduler,
                        scaler=scaler, memory=memory)
    names, pairs, retrieval = protocol_data
    images = args.project / "data/NIH_ChestXray14/images"
    for epoch in range(metadata["epoch"], epochs):
        if is_verification:
            positives = [pair for pair in pairs["train"] if pair[2] == 1]
            current = randomized_pairs(positives, names["train"], 2*len(positives), seed=args.seed + epoch)
            dataset = PairDataset(images, current, condition_id=args.condition_id)
            order = np.random.RandomState(args.seed + epoch).permutation(len(dataset)).tolist()
        else:
            dataset = RetrievalDataset(images, retrieval["train"], condition_id=args.condition_id,
                                       augment=True, seed=args.seed + epoch * 100000)
            order = mining_order(retrieval["train"], seed=args.seed + epoch)
        offset = metadata["batch"] if metadata["epoch"] == epoch else 0
        batches = loader(dataset, args, order=order, offset=offset, seed=args.seed + epoch)
        model.train()
        if memory:
            memory.train()
        accumulated = metadata["train_sum"] if offset else 0.
        sample_count = metadata["train_samples"] if offset else 0
        for batch_number, batch in enumerate(batches, start=offset):
            if (args.output / "STOP_REQUESTED").exists():
                save_checkpoint(last, model, optimizer, metadata=metadata, scheduler=scheduler, scaler=scaler, memory=memory)
                raise KeyboardInterrupt("사용자가 학습 중단을 요청했습니다.")
            first_step = metadata["global_step"] == 0
            before = parameters_digest(model, head_only=True) if first_step else None
            tensors = [item.to(args.device, non_blocking=True) for item in batch]
            label = tensors[-1]

            def forward_batch():
                with torch.autocast(args.device, dtype=torch.float16,
                                    enabled=(not is_verification and args.device == "cuda")):
                    if is_verification:
                        return F.binary_cross_entropy_with_logits(model(tensors[0], tensors[1]).flatten(), label)
                    return memory(model(tensors[0]), label)

            def record_overflow(event):
                progress(args.output, phase=phase, status="amp_backoff", epoch=epoch + 1, epochs=epochs,
                         step=batch_number + 1, loss=None, **event)

            loss, precision = optimizer_batch(model, optimizer, scaler, forward_batch,
                                               memory=memory, on_overflow=record_overflow)
            metadata["amp_overflows"] = metadata.get("amp_overflows", 0) + precision["amp_overflows"]
            if scheduler:
                scheduler.step()
            if first_step:
                atomic_json(folder / "first_backward.json", {"global_step_before": 0,
                    **precision, "head_before_sha256": before,
                    "head_after_sha256": parameters_digest(model, head_only=True), "batch": len(label),
                    "loss": float(loss.detach()), "device": args.device})
            sample_count += len(label)
            accumulated += float(loss.detach()) * len(label)
            metadata.update(epoch=epoch, batch=batch_number + 1, global_step=metadata["global_step"] + 1,
                            train_sum=accumulated, train_samples=sample_count)
            if metadata["global_step"] % args.checkpoint_every_steps == 0:
                save_checkpoint(last, model, optimizer, metadata=metadata, scheduler=scheduler, scaler=scaler, memory=memory)
            if batch_number % 25 == 0 or batch_number + 1 == full_batches:
                progress(args.output, phase=phase, status="training", epoch=epoch + 1, epochs=epochs,
                         step=batch_number + 1, steps=full_batches, global_step=metadata["global_step"],
                         loss=float(loss.detach()), learning_rate=optimizer.param_groups[0]["lr"],
                         samples_seen=sample_count, batch_size=32)
        progress(args.output, phase=phase, status="validating", epoch=epoch + 1, epochs=epochs,
                 step=full_batches, loss=accumulated / max(sample_count, 1))
        scores = validation(model, phase, protocol_data, args, epoch, memory=memory)
        if is_verification:
            current_loss = scores["validation_loss"]
            if metadata["best_loss"] is None or current_loss < metadata["best_loss"]:
                metadata["best_loss"], metadata["bad_epochs"] = current_loss, 0
                atomic_torch_save(model.state_dict(), exported)
            else:
                metadata["bad_epochs"] += 1
        else:
            # 공식 검색 코드는 validation이 아닌 마지막 학습 에폭의 모델을 내보냅니다.
            atomic_torch_save(model.state_dict(), exported)
        metadata.update(epoch=epoch + 1, batch=0, train_sum=0., train_samples=0.,
                        finished=(epoch + 1 == epochs or (is_verification and metadata["bad_epochs"] >= 5)))
        save_checkpoint(last, model, optimizer, metadata=metadata, scheduler=scheduler, scaler=scaler, memory=memory)
        atomic_torch_save(torch.load(last, map_location="cpu", weights_only=True), folder / f"epoch_{epoch+1:03d}.pt")
        progress(args.output, phase=phase, status="epoch_complete", epoch=epoch + 1, epochs=epochs,
                 step=full_batches, loss=accumulated / max(sample_count, 1), **scores)
        if metadata["finished"]:
            break
    del model, optimizer, scheduler, scaler, memory
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return exported


def preflight(args):
    seven = Path(__file__).resolve().parents[2]
    assert_training_scope(str(seven), str(args.output), str(args.project))
    official = seven / "official_source"
    manifest_audit = audit_official_manifests(official)
    image_audit = audit_image_presence(args.project, official)
    protocol_data = load_protocol(official)
    run_binding, details = binding(args, official)
    names, pairs, retrieval = protocol_data
    images = args.project / "data/NIH_ChestXray14/images"
    required = set(names["train"] + names["validation"])
    required.update(name for phase in ("train", "validation") for name, _ in retrieval[phase])
    missing = sum(not (images / name).is_file() for name in required)
    summary = {"binding": run_binding, "protocol": details, "required_training_validation_images": len(required),
               "missing_training_validation_images": missing,
               "verification_counts": {key: len(value) for key, value in names.items()},
               "retrieval_counts": {key: len(value) for key, value in retrieval.items()},
               "verification_official_pair_counts":{key:len(value) for key,value in pairs.items()},
               "verification_rnp_pair_counts":{key:2*sum(label==1 for _,_,label in value) for key,value in pairs.items() if key!='test'},
               "manifest_audit": manifest_audit, "image_audit": image_audit,
               "cleaning_policy": "UNMODIFIED_AUTHOR_SPLITS",
               "test_images_used_for_training": False}
    atomic_json(args.output / "preflight.json", summary)
    if missing:
        raise FileNotFoundError(f"학습·검증에 필요한 원본 영상 중 {missing}장이 없습니다. 다운로드 검증 완료 후 실행하세요.")
    return protocol_data, run_binding, summary


def train(args):
    args.project = args.project.resolve()
    args.output = args.output.resolve()
    seven = Path(__file__).resolve().parents[2]
    assert_training_scope(str(seven), str(args.output), str(args.project))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training_config.json").exists() and not args.resume:
        raise FileExistsError("기존 학습 실행이 있습니다. 새 output을 사용하거나 해당 실행에 --resume를 명시하세요.")
    protocol_data, run_binding, report = preflight(args)
    if args.preflight:
        progress(args.output, phase="preflight", status="ready", epoch=0, epochs=0, step=0, loss=None)
        return report
    identity = training_identity(args.condition_id, args.output.parent.name)
    config = {**identity, "binding": run_binding, "protocol_fingerprint": run_binding,
              "protocol": PROTOCOL, "seed": args.seed,
              "tasks": args.tasks, "verification_workers": args.verification_workers,
              "retrieval_workers": args.retrieval_workers,
              "activation_checkpointing": args.activation_checkpointing, "device": args.device,
              "torch": torch.__version__, "new_cxr_training": True}
    if args.resume:
        saved = json.loads((args.output / "training_config.json").read_text(encoding="utf-8"))
        validate_training_identity(saved, args.condition_id, args.output.parent.name)
        if saved != config:
            raise ValueError("재개 설정이 해당 실행의 원본 설정과 다릅니다.")
    else:
        atomic_json(args.output / "training_config.json", config)
    init, provenance = imagenet_initialization(args.imagenet_cache)
    artifacts = {**identity, "protocol_fingerprint": run_binding,
                 "kind": "조건별 새로 학습한 CXR 모델", "published_cxr_initialization": False,
                 "initialization": provenance, "binding": run_binding}
    if args.tasks in ("both", "verification"):
        path = phase_train(args, "verification", protocol_data, provenance, init, run_binding)
        artifacts.update(verification_checkpoint=str(path), verification_sha256=digest(path))
    if args.tasks in ("both", "retrieval"):
        intermediate = phase_train(args, "retrieval_phase1", protocol_data, provenance, init, run_binding)
        path = phase_train(args, "retrieval_phase2", protocol_data, provenance, init, run_binding, previous_weights=intermediate)
        artifacts.update(retrieval_checkpoint=str(path), retrieval_sha256=digest(path))
    atomic_json(args.output / "artifacts.json", artifacts)
    progress(args.output, phase="complete", status="completed", epoch=0, epochs=0, step=0, loss=None)
    return artifacts


def smoke_evaluation_forward(model, images, *, task):
    """Exercise the original full-precision evaluation shape without fallback.

    Lightning's mixed-precision validation-loss batches are not the original
    PML BaseTester pass: that separate full-set embedding pass is FP32. Its
    peak memory therefore needs an independent GPU check before a long run.
    """
    if task not in ("verification", "retrieval"):
        raise ValueError("Unsupported evaluation smoke task")
    size = 256 if task == "verification" else 1024
    output_size = 1 if task == "verification" else 128
    if tuple(images.shape) != (32, 3, size, size) or images.dtype != torch.float32:
        raise ValueError("Evaluation smoke requires the original batch 32, resolution and FP32 inputs")
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(images.device.type, enabled=False):
            output = model(images, images.roll(1, dims=0)) if task == "verification" else model(images)
        if output.dtype != torch.float32 or tuple(output.shape) != (32, output_size):
            raise RuntimeError("Evaluation smoke did not produce the expected FP32 output")
        if not torch.isfinite(output).all():
            raise FloatingPointError("Evaluation smoke output contains NaN or infinity")
        return {"evaluation_batch_size": 32, "evaluation_resolution": size,
                "evaluation_precision": "float32", "evaluation_output_shape": list(output.shape)}
    finally:
        model.train(was_training)


def smoke_gpu(args):
    """원 배치·해상도의 실제 optimizer step과 FP32 평가 경로를 검사합니다."""
    seven = Path(__file__).resolve().parents[2]
    assert_training_scope(str(seven), str(args.output), str(args.project))
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    results = []
    for phase in ("verification", "retrieval_phase1", "retrieval_phase2"):
        seed_all(args.seed, phase=phase)
        memory_gate(args, phase)
        torch.cuda.reset_peak_memory_stats()
        model = create_model("verification" if phase == "verification" else "retrieval",
                             activation_checkpointing=args.activation_checkpointing).cuda().train()
        if phase == "retrieval_phase1":
            for parameter in model[0].parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001) if phase == "verification" else torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=.00633956, momentum=.9)
        memory = CrossBatchContrastive().cuda()
        use_amp = phase != "verification"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        size = 256 if phase == "verification" else 1024
        image = torch.randn(32, 3, size, size, device="cuda")
        label = torch.arange(32, device="cuda") // 8
        before = parameters_digest(model, head_only=True)
        fresh_optimizer_entries = len(optimizer.state)
        started = time.monotonic()

        def forward_smoke():
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if phase == "verification":
                    predicted = model(image, image.roll(1, dims=0)).flatten()
                    return F.binary_cross_entropy_with_logits(predicted, (torch.arange(32, device="cuda") % 2).float())
                return memory(model(image), label)

        def record_overflow(event):
            progress(args.output, phase=phase, status="amp_backoff", epoch=0, epochs=0, step=0, loss=None, **event)

        smoke_attempts = 0
        while True:
            loss, precision = optimizer_batch(model, optimizer, scaler, forward_smoke,
                                              memory=memory if use_amp else None, on_overflow=record_overflow)
            smoke_attempts += 1
            if precision["optimizer_applied"]:
                break
            if smoke_attempts >= 32:
                raise FloatingPointError("Smoke test could not obtain one applied optimizer step")
        torch.cuda.synchronize()
        training_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.reset_peak_memory_stats()
        evaluation = smoke_evaluation_forward(model, image, task="verification" if phase == "verification" else "retrieval")
        torch.cuda.synchronize()
        evaluation_peak = torch.cuda.max_memory_allocated() / 2 ** 30
        result = {"phase": phase, "batch_size": 32, "resolution": size, "loss": float(loss.detach()),
                  **precision, "smoke_batches_attempted": smoke_attempts,
                  **evaluation,
                  "initial_optimizer_state_entries": fresh_optimizer_entries,
                  "global_step_before": 0, "optimizer_state_entries": len(optimizer.state),
                  "head_before_sha256": before, "head_after_sha256": parameters_digest(model, head_only=True),
                  "head_changed": before != parameters_digest(model, head_only=True),
                  "training_peak_gpu_gib": training_peak,
                  "evaluation_peak_gpu_gib": evaluation_peak,
                  "peak_gpu_gib": max(training_peak, evaluation_peak),
                  "seconds": time.monotonic() - started, "activation_checkpointing": args.activation_checkpointing,
                  "data_type": "합성 난수 영상; CXR 성능 평가나 정식 학습 아님"}
        results.append(result)
        atomic_json(args.output / "smoke.json", results)
        progress(args.output, phase=phase, status="smoke_pass", epoch=0, epochs=0, step=1, **{
            key: value for key, value in result.items() if key != "phase"})
        del model, optimizer, memory, scaler, image, label, loss
        gc.collect()
        torch.cuda.empty_cache()
    return results
