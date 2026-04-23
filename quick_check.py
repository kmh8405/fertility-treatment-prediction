import pandas as pd

train = pd.read_csv("data/raw/train.csv")

# 1. 데이터 크기
print("shape:", train.shape)

# 2. 기본 정보
print("\ninfo:")
train.info()
print("\ndescribe:")
train.describe()

# 3. 타겟 컬럼 설정
TARGET = "임신 성공 여부"

# 4. 타겟 분포 확인
print("\n[Target Distribution]")
print(train[TARGET].value_counts())
print("\n[Target Ratio]")
print(train[TARGET].value_counts(normalize=True))

# 5. 결측치 상위 20개 확인
print("\n[Top 20 Missing Values]")
missing = train.isnull().sum().sort_values(ascending=False)
print(missing.head(20))

# 6. 데이터 타입 분포 확인
print("\n[Data Types Count]")
print(train.dtypes.value_counts())

# 7. ID 컬럼 확인 (있으면 모델에서 제외 예정)
if "ID" in train.columns:
    print("\n[ID Column Example]")
    print(train["ID"].head())