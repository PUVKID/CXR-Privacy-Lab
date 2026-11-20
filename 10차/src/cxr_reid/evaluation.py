"""원 논문 검증·검색 지표와 확장 연구의 CMC Rank-1 보수 점수를 계산한다."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def calculate_verification_metrics(labels, scores, threshold: float = 0.5) -> dict:
    """저자 검증 과제와 동일한 0.5 판정 임계값을 사용한다."""
    truth = np.asarray(labels, dtype=np.int32)
    probability = np.asarray(scores, dtype=np.float64)
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("검증 정답과 예측값의 길이가 올바르지 않습니다.")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("동일 환자 라벨은 0 또는 1이어야 합니다.")
    if not np.isfinite(probability).all():
        raise ValueError("검증 예측에 NaN 또는 무한대가 있습니다.")
    prediction = (probability > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    total_pos = tp + fn
    total_neg = tn + fp
    result = {
        "n_pairs": len(truth), "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "far": float(fp / total_neg) if total_neg else math.nan,
        "frr": float(fn / total_pos) if total_pos else math.nan,
        "specificity": float(tn / total_neg) if total_neg else math.nan,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold),
        "eer_method":"관측 ROC의 |FNR-FPR| 최소점 평균 근사; 원 논문 기본 지표의 직접 재현 아님",
    }
    if len(np.unique(truth)) > 1:
        fpr, tpr, thresholds = roc_curve(truth, probability)
        fnr = 1 - tpr
        cross = np.argmin(np.abs(fnr - fpr))
        result["auc"] = float(roc_auc_score(truth, probability))
        result["eer"] = float((fnr[cross] + fpr[cross]) / 2)
    else:
        result["auc"] = math.nan
        result["eer"] = math.nan
    return result


def calculate_retrieval_metrics(
    query_embeddings, gallery_embeddings, query_ids, gallery_ids, *,
    same_source: bool = False, block_size: int = 128, gallery_keys=None,
    progress_callback=None,
) -> dict:
    """전체 갤러리에 대한 자기 이미지 제외 최근접 이웃·mAP@R·R-Precision을 계산한다.

    공식 PML 0.9.94 BaseTester처럼 양쪽 출력을 L2 정규화한다. 영벡터는
    sklearn.normalize와 동일하게 0으로 유지하고 개수를 기록한다. 비교할 다른
    촬영본이 존재하지 않는 singleton 환자는 식별 정확도 분모에서 제외한다.
    """
    q = np.asarray(query_embeddings, dtype=np.float32)
    g = np.asarray(gallery_embeddings, dtype=np.float32)
    qi = np.asarray(query_ids).astype(str)
    gi = np.asarray(gallery_ids).astype(str)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError("임베딩은 동일한 차원의 2차원 행렬이어야 합니다.")
    if len(q) != len(qi) or len(g) != len(gi):
        raise ValueError("임베딩과 환자 라벨 개수가 일치하지 않습니다.")
    if same_source and len(q) != len(g):
        raise ValueError("자기 이미지 제외 모드에서는 질의와 갤러리 길이가 같아야 합니다.")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("임베딩에 NaN 또는 무한대가 있습니다.")
    if block_size < 1:
        raise ValueError("block_size는 1 이상이어야 합니다.")
    from sklearn.preprocessing import normalize
    zero_queries=int(np.count_nonzero(np.linalg.norm(q,axis=1)==0))
    zero_gallery=int(np.count_nonzero(np.linalg.norm(g,axis=1)==0))
    q=normalize(q,norm='l2',axis=1,copy=True)
    g=normalize(g,norm='l2',axis=1,copy=True)
    keys=np.asarray(gallery_keys if gallery_keys is not None else [f'{i:012d}' for i in range(len(g))]).astype(str)
    if len(keys)!=len(g) or len(set(keys))!=len(keys):raise ValueError('갤러리 동률 키는 영상별 고유값이어야 합니다.')

    count_g = dict(zip(*np.unique(gi, return_counts=True)))
    top1_total = 0
    map_sum = 0.0
    rp_sum = 0.0
    valid = 0
    tied_queries=0;expected_top1=0.0
    g_norm = np.sum(g * g, axis=1)
    if progress_callback is not None:
        progress_callback(0, len(q))
    for start in range(0, len(q), block_size):
        stop = min(start + block_size, len(q))
        group = q[start:stop]
        d_sq = np.maximum(
            np.sum(group * group, axis=1)[:, None] + g_norm[None, :] - 2.0 * (group @ g.T), 0
        )
        if same_source:
            d_sq[np.arange(len(group)), np.arange(start, stop)] = np.inf
        for offset, distances in enumerate(d_sq):
            row = start + offset
            relevant = int(count_g.get(qi[row], 0)) - int(same_source and qi[row] == gi[row])
            if relevant == 0:
                continue
            valid += 1
            # R 경계의 동률 후보를 모두 포함한 뒤 영상 파일명 전순서로 결정합니다.
            cutoff=np.partition(distances,relevant-1)[relevant-1]
            candidates=np.flatnonzero(distances<=cutoff)
            nearest=candidates[np.lexsort((keys[candidates],distances[candidates]))][:relevant]
            tied=np.flatnonzero(distances==distances[nearest[0]])
            tied_queries+=int(len(tied)>1)
            expected_top1+=float(np.mean(gi[tied]==qi[row]))
            is_positive = (gi[nearest] == qi[row]).astype(np.float32)
            top1_total += int(gi[int(nearest[0])] == qi[row])
            rp_sum += float(np.mean(is_positive))
            map_sum += float(np.sum(is_positive * (np.cumsum(is_positive) / np.arange(1, relevant + 1))) / relevant)
        if progress_callback is not None:
            progress_callback(stop, len(q))
    if valid == 0:
        raise ValueError("동일 환자의 다른 촬영본이 없어 검색 평가가 불가능합니다.")
    rank_one = top1_total / valid
    return {
        "n_queries": int(len(q)), "eligible_queries": int(valid),
        "n_gallery_images":len(g),"n_gallery_patients":len(count_g),
        "aggregation_unit":"R>0인 질의 영상별 평균; 환자별 균등 평균 아님",
        # 1위 검색 정답률은 이 환자-ID 갤러리 평가 방식에서 CMC Rank-1과 동일하다.
        # 지표 자체는 반환하되 프라이버시 변환은 2·3단계에서만 명시적으로 적용한다.
        "precision_at_1": float(rank_one), "cmc_rank_1": float(rank_one),
        "map_at_r": float(map_sum / valid),
        "r_precision": float(rp_sum / valid),
        "normalization":"sklearn L2, 영벡터는 0 유지",
        "zero_query_embeddings":zero_queries,"zero_gallery_embeddings":zero_gallery,
        "tie_rule":"제곱 L2 거리 → 고유 영상 파일명 오름차순; 이름 미제공 시 입력 인덱스",
        "rank1_tied_queries":tied_queries,
        "tie_random_expected_rank1":expected_top1/valid,
    }


def calculate_privacy_score_rank1(cmc_rank1: float) -> float:
    """CMC-R1 공격 성공률의 보수(0~100점)를 확장 연구의 지표로 정의한다.

    학술 근거: Tian et al., ICCV 2025, 'Semantic versus Identity: A
    Divide-and-Conquer Approach towards Adjustable Medical Image De-Identification'
    는 흉부 X선 등의 재식별 위험 측정에 CMC-R1을 사용한다.

    PrivacyScore@1 = 100 × (1 − CMC@1)은 이 연구가 정의한 변환식이다.
    Tian 등의 논문이 동일한 이름/수식의 점수를 발표했다는 뜻은 아니다.
    인증된 비식별화나 차등프라이버시 보장을 제공하지 않는다.
    """
    try:
        success_rate = float(cmc_rank1)
    except (TypeError, ValueError) as error:
        raise ValueError("CMC@1은 0과 1 사이의 유한한 수여야 합니다.") from error
    if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
        raise ValueError("CMC@1은 0과 1 사이의 유한한 수여야 합니다.")
    return float(100.0 * (1.0 - success_rate))


def bootstrap_auc_ci(labels, scores, repeats: int, *, seed: int = 3,
                     progress_callback=None) -> dict:
    """원 논문에서 사용한 반복 복원추출 방식의 AUC 신뢰구간을 계산한다."""
    if repeats <= 0:
        return {}
    truth = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float32)
    rng = np.random.default_rng(seed)
    results = []
    if progress_callback is not None:
        progress_callback(0, repeats)
    for iteration in range(repeats):
        samples = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[samples])) == 2:
            results.append(float(roc_auc_score(truth[samples], score[samples])))
        if progress_callback is not None and ((iteration + 1) % 50 == 0 or iteration + 1 == repeats):
            progress_callback(iteration + 1, repeats)
    if not results:
        raise ValueError("AUC 부트스트랩 표본에 두 클래스가 모두 존재하지 않습니다.")
    return {
        "auc_ci_lower": float(np.quantile(results, 0.025)),
        "auc_ci_upper": float(np.quantile(results, 0.975)),
        "auc_bootstrap_mean": float(np.mean(results)),
        "bootstrap_repeats": len(results),
        "bootstrap_attempts": repeats,
        "auc_ci_method":"95% percentile pair bootstrap; 전체 쌍을 복원추출; 환자 군집 CI 아님; 저자 마지막 표본 제외 결함 복제 안 함",
    }
