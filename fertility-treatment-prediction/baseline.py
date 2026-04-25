# baseline.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# 1. 데이터 로드
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
submission = pd.read_csv("data/raw/sample_submission.csv")

# 2. 타겟 설정
TARGET = "임신 성공 여부"

# 3. ID 제거
train = train.drop(columns=["ID"])
test_ids = test["ID"]
test = test.drop(columns=["ID"])

# 4. 타겟 분리
X = train.drop(columns=[TARGET])
y = train[TARGET]

# =========================
# 🔥 1. 문자열 숫자 컬럼 처리 (핵심)
# =========================
for col in X.columns:
    if "횟수" in col:
        X[col] = X[col].astype(str).str.extract('(\d+)').astype(float)
        test[col] = test[col].astype(str).str.extract('(\d+)').astype(float)

# =========================
# 🔥 2. 나이 컬럼 처리
# =========================
age_cols = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"]

for col in age_cols:
    if col in X.columns:
        X[col] = X[col].astype(str).str.extract('(\d+)').astype(float)
        test[col] = test[col].astype(str).str.extract('(\d+)').astype(float)

# =========================
# 🔥 3. 결측치 90% 이상 컬럼 제거
# =========================
missing_ratio = X.isnull().mean()
drop_cols = missing_ratio[missing_ratio > 0.9].index

X = X.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# =========================
# 5. 컬럼 타입 재분리 (위에서 바뀌었기 때문)
# =========================
cat_cols = X.select_dtypes(include=["object", "string"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# =========================
# 6. 결측치 처리 (train 기준)
# =========================
num_imputer = SimpleImputer(strategy="constant", fill_value=-1)
X[num_cols] = num_imputer.fit_transform(X[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])

cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])

# =========================
# 7. 범주형 인코딩 (train 기준)
# =========================
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X[cat_cols] = encoder.fit_transform(X[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

# =========================
# 8. 모델 학습
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# =========================
# 9. 예측 (확률)
# =========================
pred = model.predict_proba(test)[:, 1]

# =========================
# 10. 제출 파일 생성
# =========================
submission["probability"] = pred
print(submission.shape)
submission.to_csv("submissions/submission_v1.csv", index=False)

print("✅ submission_baseline.csv 생성 완료!")