# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:45:12 2023
"""

# 1. 데이터 전처리
import pandas as pd
import numpy as np

# (1) 데이터 로드
df = pd.read_csv("C:\서울시 지하철 30분단위 출발-도착 데이터.csv", encoding='cp949') # 디렉토리

# (2) 년도별 승하차 이용객이 가장 많은 역
crowd = df.sort_values("인원합계(PASSN_CNT)", ascending = False).drop_duplicates("년(YEAR)").sort_values("년(YEAR)")
print(crowd[["년(YEAR)", "승차_역명(GETON_STATION_NM)", "하차_역명(GETOFF_STATION_NM)", "인원합계(PASSN_CNT)"]])


# 2. 승하차 이용 인원 예측하기
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# (1) 데이터 분리
X = df.drop("인원합계(PASSN_CNT)", axis = 1)
y = df["인원합계(PASSN_CNT)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 111)

# (2) EDA
X_train.info()
X_train.head(30)
X_train.describe()

y_train.info()
y_train.head(30)
y_train.describe()

# (3) 데이터 전처리
X_train = X_train.drop(["승차역ID(GETON_STATION_ID)", "하차역ID(GETOFF_STATION_ID)"], axis = 1)
X_test = X_test.drop(["승차역ID(GETON_STATION_ID)", "하차역ID(GETOFF_STATION_ID)"], axis = 1)

label = LabelEncoder()

X_train["사용자구분(BILL_USER)"] = label.fit_transform(X_train["사용자구분(BILL_USER)"])
X_train["승차_호선명(GETON_LINE_NM)"] = label.fit_transform(X_train["승차_호선명(GETON_LINE_NM)"])
X_train["승차_역명(GETON_STATION_NM)"] = label.fit_transform(X_train["승차_역명(GETON_STATION_NM)"])
X_train["하차_호선명(GETOFF_LINE_NM)"] = label.fit_transform(X_train["하차_호선명(GETOFF_LINE_NM)"])
X_train["하차_역명(GETOFF_STATION_NM)"] = label.fit_transform(X_train["하차_역명(GETOFF_STATION_NM)"])

X_test["사용자구분(BILL_USER)"] = label.fit_transform(X_test["사용자구분(BILL_USER)"])
X_test["승차_호선명(GETON_LINE_NM)"] = label.fit_transform(X_test["승차_호선명(GETON_LINE_NM)"])
X_test["승차_역명(GETON_STATION_NM)"] = label.fit_transform(X_test["승차_역명(GETON_STATION_NM)"])
X_test["하차_호선명(GETOFF_LINE_NM)"] = label.fit_transform(X_test["하차_호선명(GETOFF_LINE_NM)"])
X_test["하차_역명(GETOFF_STATION_NM)"] = label.fit_transform(X_test["하차_역명(GETOFF_STATION_NM)"])

# (4) 테스트/검증용 데이터 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 111)

# (5) 모델링
lr = LinearRegression()
lr.fit(X_tr, y_tr)
pred = lr.predict(X_val)
print("Linear Regression - RMSE :", np.sqrt(mean_squared_error(y_val, pred)))

svm = SVR()
svm.fit(X_tr, y_tr)
pred = svm.predict(X_val)
print("SVM - RMSE :", np.sqrt(mean_squared_error(y_val, pred)))

dt = DecisionTreeRegressor()
dt.fit(X_tr, y_tr)
pred = dt.predict(X_val)
print("Decision Tree - RMSE :", np.sqrt(mean_squared_error(y_val, pred)))

rf = RandomForestRegressor()
rf.fit(X_tr, y_tr)
pred = rf.predict(X_val)
print("Random Forest - RMSE :", np.sqrt(mean_squared_error(y_val, pred)))

xgb = XGBRegressor()
xgb.fit(X_tr, y_tr)
pred = xgb.predict(X_val)
print("XGBoost - RMSE :", np.sqrt(mean_squared_error(y_val, pred)))

# (6) 최종 모형 선정
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# (7) 평가
print("RMSE -", mean_squared_error(y_test, prediction) ** 0.5)

result = X_test
result["예상 이용 인원"] = prediction
result["실제 이용 인원"] = y_test
print(result.head(5))
result.to_csv("20230110.csv", encoding='utf-8-sig', index = False)


# 결론: 실제 적용이 불가능할 정도로 모델의 정확도가 현저히 떨어짐
# (500개로 제한된 샘플 데이터의 한계가 원인으로 추측, 또는 변수 선택 문제)
