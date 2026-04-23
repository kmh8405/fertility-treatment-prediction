# 🧬 Fertility Treatment Prediction (IVF Success Prediction)

현재 난임 환자 시술 데이터를 기반으로 임신 성공 여부를 예측하는 머신러닝 모델을 개발 중입니다.  
해커톤 참여 프로젝트이며, ROC-AUC를 기준으로 성능을 평가합니다.

---

# 📌 Project Goal

- IVF 기반 난임 시술 데이터 분석 및 예측 모델 개발
- 목표: 임신 성공 여부 (0/1) 예측
- 평가 지표: ROC-AUC

---

# 📊 Current Status (Work in Progress)

현재 프로젝트 초기 단계이며 아래 작업이 진행 중입니다.

- [x] 데이터 구조 확인 및 초기 분석
- [x] baseline 모델 구성 및 1차 제출 준비
- [ ] 팀원별 EDA 병렬 진행
- [ ] EDA 결과 통합 (eda_final.ipynb)
- [ ] Feature Engineering
- [ ] 모델 개선 및 검증

---

# ⚙️ Workflow

- 각자 EDA 브랜치 생성 후 분석 진행
- PR 기반으로 main 브랜치에 통합
- 이후 feature engineering 및 모델링 단계로 확장

---

# 📂 Notes

- 데이터 및 제출 파일은 GitHub에 업로드하지 않음 (.gitignore 적용)
- 모든 전처리 및 모델링은 train 데이터를 기준으로 진행 (data leakage 방지)
- 현재 구조는 초기 설정 단계이며 이후 리팩토링 예정