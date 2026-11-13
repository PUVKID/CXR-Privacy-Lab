# 학습 포팅의 출처와 라이선스

이 폴더는 다음 공식 소스와 논문에 근거한 현대 PyTorch 학습 포팅입니다.

- 원 저장소: https://github.com/kaipackhaeuser/CXR-Patient-ReID
- 고정 커밋: `880168f564afa07936a8d8fb3b868b140ab937d9`
- 원 논문: Packhäuser et al., Scientific Reports 12, 14851 (2022), DOI https://doi.org/10.1038/s41598-022-19045-3
- 공식 `PatientRetrieval/paper_retrieval_training_phase1.py`, `paper_retrieval_training_phase2.py`의 `sample_func`, `get_block`, `build_mining_list_32`에 따른 블록8 샘플러와 모델·학습 조건을 `data.py`, `engine.py`로 이식했습니다. 공식 `PatientVerification`의 RNP 데이터 구성과 학습 조건도 이식했습니다.
- 원 공식 저장소의 GNU GPL v3 라이선스 전문을 `LICENSE`에 변경 없이 보존했습니다. 해당 파생 부분의 기존 라이선스·출처를 제거하거나 임의의 다른 라이선스로 변경하지 않습니다. 원본 소스의 한국어 주석 이식과 현대 API·안전한 저장·재개·입력 경계 검사는 이번 차수의 변경입니다.
- PML 0.9.94의 `ContrastiveLoss`, `AvgNonZeroReducer`, `BaseDistance`, `CrossBatchMemory` 동작에 근거해 정규화 유클리드 거리·margin0/1·양/음성별 비영 손실 평균·메모리128을 구현했습니다. 출처: https://github.com/KevinMusgrave/pytorch-metric-learning/tree/v0.9.94 . 이 저장소의 MIT 라이선스와 Copyright (c) 2019 Kevin Musgrave는 `LICENSE_PML`에 보존합니다.

이 안내는 의료 원영상·가명 환자별 목록·훈련 가중치의 공개 허가가 아닙니다. 기존 첨부 평가 패키지의 재배포 권한 판단과 이 학습 포팅의 파생 소스 의무는 별도 검토 대상입니다. 학습 모듈이 참조하는 `cxr_reid` 패키지를 포함한 전체 배포의 공개 가능 범위는 부모의 독립 검토 전 확정하지 않습니다.
