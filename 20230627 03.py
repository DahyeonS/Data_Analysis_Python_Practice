# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:41:32 2023

@author: 서다현
"""

'''
이용 데이터: 이직 여부 판단 데이터
작업 유형: 이직 여부 예측(분류)
사용 모델: 텐서플로우 기반 인공신경망
'''

# 1. 패키지 임포트
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# 2. 데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_test.csv")

# 3. 데이터 전처리
# (1) 사용 데이터 선택
X = x_train.drop("StudentID", axis = 1)
y = y_train.drop("StudentID", axis = 1)

x_test_id = x_test.pop("StudentID")

# (2) EDA
X.info()
X.describe().iloc[2,:] # absences, G1, G2의 표준편차가 1에서 멂

# (3) 변수 선택
X_corr = X.corr().unstack().reset_index().dropna()
X_corr.sort_values(0, ascending = False).drop_duplicates(0)[X_corr[0] != 1].head(5).reset_index(drop = True)
'''
G1 - G2는 상관계수가 0.7 이상임
Dalc - Walc, Fedu - Medu는 상관계수가 0.5 이상임
G2, Walc, Medu는 변수 제외
'''
X = X.drop("G2", axis = 1)
X = X.drop("Walc", axis = 1)
X = X.drop("Medu", axis = 1)

x_test = x_test.drop("G2", axis = 1)
x_test = x_test.drop("Walc", axis = 1)
x_test = x_test.drop("Medu", axis = 1)

# (4) 정규화
scale = MinMaxScaler()

target = ["absences", "G1"]
X[target] = scale.fit_transform(X[target])

# (5) 범주형 변수 변환
X = pd.get_dummies(X)
x_test = pd.get_dummies(x_test)

# 4. 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.3, random_state = 333)

# 5. 모델링
model = Sequential()

model.add(Dense(20, input_dim = 55, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1))
model.summary()

model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(X_tr, y_tr, epochs = 100, verbose = 1, validation_data=(X_val, y_val))

# 6. 평가
loss_val, mae = model.evaluate(X_val, y_val)
print('loss value =', loss_val)
print('mae =', mae)


# 7. 적용
model.fit(X, y, epochs = 100, verbose = 1)
y_pred = model.predict(x_test)

y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_test.csv")
y_test = y_test.iloc[:,1]

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

predict = pd.DataFrame(y_pred)[0]
result = pd.DataFrame({"StudentID":x_test_id, "G3":predict})
result.to_csv("20230627 03.csv", index = False)

pd.read_csv("20230627 03.csv")
