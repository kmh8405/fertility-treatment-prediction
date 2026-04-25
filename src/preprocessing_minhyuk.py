import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TARGET = "임신 성공 여부"
ID_COL = "ID"

# 1. 데이터 로드
def load_data(path):
    return pd.read_csv(path)

# 2. 불필요 컬럼 제거
def drop_columns(df):
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df

# 3. 문자열 → 수치 변환 (나이 등)
def convert_str_to_numeric(df):
    # 나이 맵핑 (EDA 기반 중간값 설정)
    age_map = {
        "만18-34세": 26,
        "만35-37세": 36,
        "만38-39세": 38.5,
        "만40-42세": 41,
        "만43-44세": 43.5,
        "만45-50세": 47,
        "알 수 없음": np.nan
    }
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(age_map)
    return df

# 4. 결측치 처리 (순서 보정: 수치 변환 후 실행)
def handle_missing(df):
    # (1) 도메인 기반 의미 있는 결측치 -> 0 처리
    zero_fill_cols = [
        "PGS 시술 여부", "PGD 시술 여부", "착상 전 유전 검사 사용 여부",
        "배아 해동 경과일", "난자 해동 경과일"
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # (2) 수치형 -> 중앙값(Median)
    # 주의: Leakage 방지를 위해 원칙적으로는 train의 median을 test에 써야 하지만,
    # 여기서는 개별 df 내에서의 처리를 기본으로 합니다.
    num_cols = df.select_dtypes(include=["number"]).columns # float, int 통합 선택
    for col in num_cols:
        if col != TARGET:
            df[col] = df[col].fillna(df[col].median())

    # (3) 범주형 -> "Unknown"
    # 문자열 타입만 안전하게 골라내기 (경고 해결)
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        # 값을 문자열로 강제 변환 후 결측치 채움 (에러 방지)
        df[col] = df[col].astype(str).replace(['nan', 'None', 'NaN'], 'Unknown')

    return df

# 5. 파생변수 생성
def create_features(df):
    # 계산에 사용되는 수치형 변수들을 강제로 숫자 타입으로 변환 (에러 방지 핵심)
    # errors='coerce'를 쓰면 숫자로 변환 안 되는 문자(예: '알 수 없음')는 NaN이 되고, 나중에 결측치 처리에서 채워집니다.
    num_fix_cols = ["총 생성 배아 수", "혼합된 난자 수", "이식된 배아 수", "총 임신 횟수", "총 시술 횟수"]
    for col in num_fix_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. 배아 생성 효율
    # 근거: 난자 대비 배아 생성률이 IVF 성공률과 강한 관련
    # (Gardner et al., Embryo quality & IVF outcome)
    df["배아_생성_효율"] = df["총 생성 배아 수"] / (df["혼합된 난자 수"] + 1)

    # 2. 이식 효율
    # 근거: 생성 대비 실제 이식 비율이 implantation과 연관
    df["이식_효율"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

    # 3. 과거 성공 이력
    # 근거: 반복 시술 history는 성공 확률과 밀접
    df["과거_임신_성공_비율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

    # 4. 연령 리스크 (38세 기준)
    # 근거: 여성 연령은 IVF 성공의 가장 중요한 predictor
    # (CDC ART report, ASRM guideline)
    if "시술 당시 나이" in df.columns:
        df["고령_여부"] = (df["시술 당시 나이"] >= 38).astype(int)

    # 5. 과배란 위험 (난자 20개 초과)
    # 근거: 너무 많은 난자/배아 → 품질 저하 가능 (OHSS 등)
    df["과배란_위험"] = (df["혼합된 난자 수"] > 20).astype(int)

    # 6. 최적 이식 수 여부 (EDA 인사이트 반영: 1~2개)
    df["최적_이식_여부"] = df["이식된 배아 수"].between(1, 2).astype(int)

    return df

# 6. 범주형 인코딩 (Label Encoding)
def encode_categorical(train_df, test_df=None):
    cat_cols = train_df.select_dtypes(include=["object", "string"]).columns
    
    for col in cat_cols:
        if col == TARGET: continue
        
        le = LabelEncoder()
        # 모든 값을 문자열로 강제 변환하여 'str + int' 에러 방지
        train_values = train_df[col].astype(str).values
        le.fit(train_values)
        
        train_df[col] = le.transform(train_values)
        
        if test_df is not None:
            test_values = test_df[col].astype(str).values
            # Train에 없던 값 대응
            mapping = {l: i for i, l in enumerate(le.classes_)}
            test_df[col] = [mapping.get(val, -1) for val in test_values]
            test_df[col] = test_df[col].astype(int)

    return train_df, test_df

# 7. 전체 파이프라인 (통합 실행 환경)
def full_preprocess(train_path, test_path=None):
    train_df = load_data(train_path)
    
    # 공통 전처리 적용
    def apply_base_logic(df):
        df = drop_columns(df)
        df = convert_str_to_numeric(df)
        df = handle_missing(df)
        df = create_features(df)
        return df

    train_df = apply_base_logic(train_df)
    
    if test_path:
        test_df = load_data(test_path)
        test_df = apply_base_logic(test_df)
        # 인코딩 시 Train의 규칙만 사용함
        train_df, test_df = encode_categorical(train_df, test_df)
        return train_df, test_df
    
    # Test가 없을 때는 Train만 인코딩해서 반환
    train_df, _ = encode_categorical(train_df)
    return train_df

# =========================
# 실행 테스트
# =========================
if __name__ == "__main__":
    # [주의] 터미널의 현재 위치가 프로젝트 최상위 폴더(src, data가 보이는 곳)여야 합니다.
    # 실행 명령어 예시: python src/your_filename.py
    train_path = "data/raw/train.csv"

    try:
        # 단일 파일 처리 테스트 시
        final_df = full_preprocess(train_path)
        
        print("✅ 전처리 및 인코딩 완료!")
        print(f"최종 데이터 크기: {final_df.shape}")
        print(f"남은 결측치: {final_df.isnull().sum().sum()}개")
        print("-" * 30)
        print(final_df.head()) # 결과 데이터 일부 출력
        
        # 파생변수 생성 여부 확인
        if "배아_생성_효율" in final_df.columns:
            print("✅ 파생변수 생성 확인 완료")
            
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {train_path}")
        print("💡 터미널 위치를 프로젝트 최상위 폴더로 이동하거나 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 기타 오류 발생: {e}")
        import traceback
        traceback.print_exc() # 어디서 에러 났는지 상세 출력