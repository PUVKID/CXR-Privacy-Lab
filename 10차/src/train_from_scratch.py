"""ImageNet 초기값에서 검증·검색 CXR 모델을 새로 학습합니다."""

import argparse
from pathlib import Path
import traceback

from training.engine import train, smoke_gpu
from training.state import progress
from cxr7_contract import assert_training_scope
from cxr10_contract import condition_plan


def parse_args():
    parser = argparse.ArgumentParser(description="공개 최종 CXR 모델을 초기값으로 쓰지 않는 논문 기반 재학습")
    parser.add_argument("--project", type=Path, required=True, help="official과 data를 포함한 공유 연구 폴더")
    parser.add_argument("--output", type=Path, required=True, help="이번 UUID 전용 새 학습 결과 폴더")
    parser.add_argument("--condition-id", choices=[row['id'] for row in condition_plan()], required=True,
                        help="원영상부터 적용할 독립학습 조건")
    parser.add_argument("--tasks", choices=["both", "verification", "retrieval"], default="both")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--verification-workers", type=int, default=16)
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--resume", action="store_true", help="해당 output의 자기 학습 체크포인트만 명시적으로 재개")
    parser.add_argument("--preflight", action="store_true", help="원본 분할·학습 입력만 검사하고 학습하지 않음")
    parser.add_argument("--smoke-gpu", action="store_true", help="합성 영상으로 세 단계의 정식 배치32 역전파와 FP32 평가 경로 검사")
    parser.add_argument("--activation-checkpointing", action="store_true", help="배치32를 유지하며 재계산으로 활성값 메모리 절약")
    parser.add_argument("--no-wait-for-memory", dest="wait_for_memory", action="store_false")
    parser.add_argument("--memory-wait-seconds", type=int, default=21600)
    parser.add_argument("--imagenet-cache", type=Path,
                        default=Path(__file__).resolve().parent.parent / "work/imagenet_cache")
    args = parser.parse_args()
    if (args.seed != 3 or args.verification_workers < 0 or args.retrieval_workers < 0 or
            args.checkpoint_every_steps < 1 or args.memory_wait_seconds < 0):
        parser.error("작업자 수·체크포인트 간격·메모리 대기시간이 유효하지 않습니다.")
    return args


if __name__ == "__main__":
    args = parse_args()
    assert_training_scope(str(Path(__file__).resolve().parent.parent), str(args.output), str(args.project))
    try:
        smoke_gpu(args) if args.smoke_gpu else train(args)
    except KeyboardInterrupt as error:
        progress(args.output, phase="stopped", status="stopped", epoch=0, epochs=0, step=0, loss=None, error=str(error))
        raise SystemExit(130)
    except Exception as error:
        progress(args.output, phase="failed", status="failed", epoch=0, epochs=0, step=0, loss=None,
                 error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise SystemExit(1)
