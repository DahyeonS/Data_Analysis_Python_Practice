# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:27:50 2023
"""

# 1. 데이터 전처리
import pandas as pd

# (1) 데이터 로드
time = pd.read_csv("C:\내국인(블록) 일자별시간대별.csv", encoding = "cp949") # 디렉토리
age = pd.read_csv("C:\내국인(집계구) 성별연령대별.csv", encoding = "cp949") # 디렉토리
place = pd.read_csv("C:\내국인(집계구) 유입지별.csv", encoding = "cp949") # 디렉토리
code = pd.read_csv("C:\신한카드 내국인 63업종 코드.csv", encoding = "cp949") # 디렉토리

code["내국인업종코드(SB_UPJONG_CD)"] = code["내국인업종코드(SB_UPJONG_CD)"].str.upper()

# (2) 시간대별
df_t = pd.merge(time, code).dropna()

# 매출이 가장 많았던 시간과 가맹점
sales_t = df_t.sort_values("카드이용금액계(AMT_CORR)", ascending = False).head(1)
print(sales_t[["일별(TS_YMD)", "요일(DAW)", "시간대(TM)", "내국인업종분류(SB_UPJONG_NM)"]])

# 요일별 매출이 가장 많은 가맹점
sales_d = df_t.sort_values("카드이용금액계(AMT_CORR)", ascending = False).drop_duplicates("요일(DAW)")
print(sales_d[["일별(TS_YMD)", "요일(DAW)", "내국인업종분류(SB_UPJONG_NM)", "카드이용금액계(AMT_CORR)"]])

# 시간대별 매출 합계
print(df_t.groupby("시간대(TM)").sum()["카드이용금액계(AMT_CORR)"])

# 매출이 가장 많은 시간대 Top5
print(df_t.groupby("시간대(TM)").sum()["카드이용금액계(AMT_CORR)"].sort_values(ascending = False).head(5))

# 요일별 매출 합계
print(df_t.groupby("요일(DAW)").sum()["카드이용금액계(AMT_CORR)"].sort_values(ascending = False))

# (3) 성별연령대별
df_a = pd.merge(age, code).dropna()

# 연령별 매출이 가장 많은 가맹점
print(df_a.sort_values(["카드이용금액계(AMT_CORR)"], ascending = False).drop_duplicates("연령대별(AGE_GB)")[["연령대별(AGE_GB)", "내국인업종분류(SB_UPJONG_NM)", "카드이용금액계(AMT_CORR)"]].sort_values("연령대별(AGE_GB)"))

# 연령별 매출 합계
print(df_a.groupby("연령대별(AGE_GB)").sum()["카드이용금액계(AMT_CORR)"])

# 매출이 가장 많은 연령대 Top5
print(df_a.groupby("연령대별(AGE_GB)").sum()["카드이용금액계(AMT_CORR)"].sort_values(ascending = False)[:5])

# (4) 지역별
df_p = pd.merge(place, code).dropna()
df_p['고객주소시군구(SGG)'] = df_p['고객주소시군구(SGG)'].str.replace("강동구서울", "강동구")
df_p['고객주소광역시(SIDO)'] = df_p['고객주소광역시(SIDO)'].str.replace("서울특별시", "서울")
df_p['고객주소광역시(SIDO)'] = df_p['고객주소광역시(SIDO)'].str.replace("강원도", "강원")
df_p['고객주소광역시(SIDO)'] = df_p['고객주소광역시(SIDO)'].str.replace("경기도", "경기")

# 광역시별 매출이 가장 많은 가맹점
print(df_p.sort_values("카드이용금액계(AMT_CORR)", ascending = False).drop_duplicates("고객주소광역시(SIDO)")[["고객주소광역시(SIDO)", "내국인업종분류(SB_UPJONG_NM)", "카드이용금액계(AMT_CORR)"]])

# 광역시별 매출 합계
print(df_p.groupby("고객주소광역시(SIDO)").sum()["카드이용금액계(AMT_CORR)"])

# 카드사용이 가장 많은 광역시 Top3
print(df_p.groupby("고객주소광역시(SIDO)").sum()["카드이용건수(USECT_CORR)"].sort_values(ascending = False).head(3))

# 시군구별 카드사용이 가장 많은 가맹점
print(df_p.sort_values("카드이용건수(USECT_CORR)", ascending = False).drop_duplicates("고객주소시군구(SGG)")[["고객주소시군구(SGG)", "내국인업종분류(SB_UPJONG_NM)", "카드이용건수(USECT_CORR)"]])

# 시군구별 카드사용량 합계
print(df_p.groupby("고객주소시군구(SGG)").sum()["카드이용건수(USECT_CORR)"].sort_values(ascending = False))

# 매출이 가장 많은 시군구 Top10
print(df_p.groupby("고객주소시군구(SGG)").sum()["카드이용금액계(AMT_CORR)"].sort_values(ascending = False)[:10])

# 2. 매출 예측
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# (1) 데이터 분리
X = age.drop(["기준년월(TS_YM)", "카드이용금액계(AMT_CORR)"], axis = 1)
y = age["카드이용금액계(AMT_CORR)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)

X_train = X_train.drop("가맹점집계구코드(TOT_REG_CD)", axis = 1)
test_code = X_test.pop("가맹점집계구코드(TOT_REG_CD)")

# (2) EDA
X_train.info()
X_train.head(10)
X_train.describe()

y_train.info()

X_test.info()
y_test.info()

# (3) 데이터 정제
# 결측치 처리
X_train["성별(SEX_CCD)"] = X_train["성별(SEX_CCD)"].fillna("불명")
X_test["성별(SEX_CCD)"] = X_test["성별(SEX_CCD)"].fillna("불명")

X_train["연령대별(AGE_GB)"] = X_train["연령대별(AGE_GB)"].fillna("불명")
X_test["연령대별(AGE_GB)"] = X_test["연령대별(AGE_GB)"].fillna("불명")

# 범주형 변수 변환
X_train["내국인업종코드(SB_UPJONG_CD)"] = X_train["내국인업종코드(SB_UPJONG_CD)"].str[2:].astype(int)
X_test["내국인업종코드(SB_UPJONG_CD)"] = X_test["내국인업종코드(SB_UPJONG_CD)"].str[2:].astype(int)

label = LabelEncoder()

X_train["개인법인구분(PSN_CPR)"] = label.fit_transform(X_train["개인법인구분(PSN_CPR)"])
X_train["성별(SEX_CCD)"] = label.fit_transform(X_train["성별(SEX_CCD)"])
X_train["연령대별(AGE_GB)"] = label.fit_transform(X_train["연령대별(AGE_GB)"])

X_test["개인법인구분(PSN_CPR)"] = label.fit_transform(X_test["개인법인구분(PSN_CPR)"])
X_test["성별(SEX_CCD)"] = label.fit_transform(X_test["성별(SEX_CCD)"])
X_test["연령대별(AGE_GB)"] = label.fit_transform(X_test["연령대별(AGE_GB)"])

# 시계열 데이터 변환
X_train["일별(TS_YMD)"] = pd.to_datetime(X_train["일별(TS_YMD)"].astype(str))
X_train["년"] = X_train["일별(TS_YMD)"].dt.year
X_train["월"] = X_train["일별(TS_YMD)"].dt.month
X_train["일"] = X_train["일별(TS_YMD)"].dt.day
X_train = X_train.drop("일별(TS_YMD)", axis = 1)

X_test["일별(TS_YMD)"] = pd.to_datetime(X_test["일별(TS_YMD)"].astype(str))
X_test["년"] = X_test["일별(TS_YMD)"].dt.year
X_test["월"] = X_test["일별(TS_YMD)"].dt.month
X_test["일"] = X_test["일별(TS_YMD)"].dt.day
X_test = X_test.drop("일별(TS_YMD)", axis = 1)

# 상관계수 확인
corr = X_train.corr()
print(corr)

# 정규성 검성
import matplotlib.pyplot as plt
from scipy import stats
plt.figure()
stats.probplot(X_train["카드이용건수(USECT_CORR)"], plot = plt)
plt.show()
stats.shapiro(X_train["카드이용건수(USECT_CORR)"])

# 정규화
scale = StandardScaler()

target = ["카드이용건수(USECT_CORR)"]
X_train[target] = scale.fit_transform(X_train[target])

# (4) 테스트/검증용 데이터 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1234)

# (5) 모델링
lr = LinearRegression()
lr.fit(X_tr, y_tr)
pred = lr.predict(X_val)
print("Linear Regression -", np.sqrt(mean_squared_error(y_val, pred)))

dt = DecisionTreeRegressor()
dt.fit(X_tr, y_tr)
pred = dt.predict(X_val)
print("Decision Tree -", np.sqrt(mean_squared_error(y_val, pred)))

rf = RandomForestRegressor(min_samples_leaf = 3, max_leaf_nodes = 300, max_samples = 10, max_depth = 30)
rf.fit(X_tr, y_tr)
pred = rf.predict(X_val)
print("Random Forest -", np.sqrt(mean_squared_error(y_val, pred)))

xgb = XGBRegressor(n_estimators = 500)
xgb.fit(X_tr, y_tr)
pred = xgb.predict(X_val)
print("XGBoost -", np.sqrt(mean_squared_error(y_val, pred)))

# (6) 최종 모델 선정
model = RandomForestRegressor(min_samples_leaf = 3, max_leaf_nodes = 300, max_samples = 10, max_depth = 30)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(mean_squared_error(y_test, prediction) ** 0.5)

result = pd.DataFrame({"가맹점집계구코드":test_code, "예측금액":prediction, "실제금액":y_test})
result.head(10)
result.to_csv("20230118.csv", index = False, encoding='utf-8-sig')
