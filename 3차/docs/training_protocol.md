# 새 CXR 모델 학습과 평가 조건

<!-- reading-levels17:start -->
**읽기 수준 선택** · [LEVEL 1 - Beginner](../../docs/%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A81.md) · [LEVEL 2 - Explorer](../../docs/%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A82.md) · [LEVEL 3 - Practitioner](../../docs/%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A83.md) · [LEVEL 4 - Advanced](../../docs/%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A84.md) · [LEVEL 5 - Expert](../../docs/%EC%9D%BD%EA%B8%B0%EC%95%88%EB%82%B4/%EB%A0%88%EB%B2%A85.md)
<!-- reading-levels17:end -->

## 원 논문과 맞춘 조건

원 저자 고정 커밋 `880168f564afa07936a8d8fb3b868b140ab937d9`을 기준으로 모델 구조·학습 분할·손실과 최적화 조건을 이식합니다. 공개 CXR 최종 가중치는 학습 초기값으로 쓰지 않습니다. 원 논문의 ImageNet 초기화 파일 `resnet50-19c8e357.pth`와 새 head 초기화·새 optimizer를 사용하며 파일 SHA를 기록합니다.

<a id="layout18-table-1-row-1"></a>

**검증 모델**

**항목**

검증 모델

**조건**

입력 256, 배치 32, BCEWithLogitsLoss, Adam lr0.0001, 최대 100에폭, validation BCE 기준 patience5

<a id="layout18-table-1-row-2"></a>

**검증 쌍**

**항목**

검증 쌍

**조건**

양성 396,147쌍과 매 에폭 다시 구성한 음성 396,147쌍. validation은 별도 환자 집합에서 음성 재구성

<a id="layout18-table-1-row-3"></a>

**검증 분할**

**항목**

검증 분할

**조건**

학습 75,708장·validation10,816장·test25,596장. 검색 분할의 수량과 구분

<a id="layout18-table-1-row-4"></a>

**검색 모델**

**항목**

검색 모델

**조건**

입력 1024, 배치 32, 학습 61,755장, 환자별 블록 8 샘플러, 밝기·대비 0.4 증강

<a id="layout18-table-1-row-5"></a>

**검색 손실**

**항목**

검색 손실

**조건**

PML0.9.94 기준 정규화 L2 ContrastiveLoss, CBM128, 자기 비교 제외

<a id="layout18-table-1-row-6"></a>

**검색 최적화**

**항목**

검색 최적화

**조건**

SGD momentum0.9, weight_decay0.00001, OneCycle max_lr0.158489

<a id="layout18-table-1-row-7"></a>

**검색 단계**

**항목**

검색 단계

**조건**

출력부 30에폭 → 같은 실행의 가중치로 전체 50에폭. 각 단계의 optimizer·scheduler·CBM은 새로 구성

<a id="layout18-table-1-row-8"></a>

**선택과 평가**

**항목**

선택과 평가

**조건**

검증 모델은 validation BCE 기준, 검색은 공식 경로의 마지막 에폭. 시험 영상과 COVID를 학습·선택에 사용하지 않음


**Windows·Python3.12·PyTorch2.7.1 포팅은 논문 당시 Linux 환경과 동일하지 않습니다.** GPU 메모리를 절약하기 위한 활성값 재계산을 선택할 경우 같은 배치·난수·BatchNorm 상태와 기울기를 보존하는지 검사하고 실행 설정에 기록합니다. 배치 크기를 묵시적으로 줄이지 않습니다.

공식 검색 코드에서는 validation에서도 PML0.9.94의 CBM 손실을 호출해 메모리 큐를 갱신하는 경로가 있습니다. 이번 구현은 검증 영상이 다음 학습의 메모리에 섞이지 않도록 validation에서 큐를 갱신하지 않습니다. validation은 검색 지표를 별도로 계산하며 모델 선택에 시험 집합을 쓰지 않습니다. 이 누수 방어와 AMP 재시도·활성값 재계산은 원 공식 실행과의 의도된 차이이며, 모든 학습 동작을 그대로 복제했다고 주장하지 않습니다.

## 실행과 기록

`train_from_scratch.py --project <연구 폴더> --output <새 실행 폴더> --seed 3 --tasks both --validated-snapshot <완료 검증 스냅샷>`이 새 학습 진입점입니다. 먼저 상위 실행기의 `--prepare-only`로 데이터 준비·전수 검증을 완료하고, 그 실행이 기록한 `validation_snapshot` 경로를 전달합니다. 학습 진입점은 현재 원영상의 SHA256을 다시 대조합니다. `--preflight`는 입력 검사, `--smoke-gpu`는 합성 영상의 역전파 검사로 정식 학습과 구분합니다. 실제 학습은 데이터의 전수 무결성·환자 분리를 먼저 확인한 뒤 시작합니다.

진행 상태, 초기화 출처, optimizer와 난수 상태, 가중치 해시와 선택 근거를 실행별로 저장합니다. 재개는 동일 실행에 명시적으로 요청하고 입력·소스·설정이 일치할 때만 허용합니다. 평가에는 해당 실행이 생성한 순수 state_dict를 전달합니다.

## 해석 범위

현재 실험 범위는 NIH와 COVID-19의 재식별 및 블러·모자이크 비교입니다. CheXpert와 원 논문의 모든 부가 분석까지 포함한 완전 재현으로 표현하지 않습니다. 프라이버시 점수는 고정 공격에서의 Rank-1 실패율이며 영상의 진단 효용이나 익명성을 보장하지 않습니다. 라이선스·포팅 출처는 학습 코드 폴더의 NOTICE와 LICENSE를 확인하십시오.
