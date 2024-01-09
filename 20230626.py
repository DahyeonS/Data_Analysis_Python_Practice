# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:02:01 2023

@author: 서다현
"""

'''
이용 데이터: 성인 건강검진 데이터
작업 유형: 흡연자/비흡연자 분류
사용 모델: 텐서플로우 기반 인공신경망
'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/x_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/x_test.csv")


# 데이터 전처리
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = x_train.drop("ID", axis = 1)
y = y_train.drop("ID", axis = 1)
x_test_id = x_test.pop("ID")

label = LabelEncoder()
X["성별코드"] = label.fit_transform(X["성별코드"])
X["구강검진수검여부"] = label.fit_transform(X["구강검진수검여부"])
X["치석"] = label.fit_transform(X["치석"])
x_test["성별코드"] = label.fit_transform(x_test["성별코드"])
x_test["구강검진수검여부"] = label.fit_transform(x_test["구강검진수검여부"])
x_test["치석"] = label.fit_transform(x_test["치석"])

scale = MinMaxScaler()
target = ["체중(5Kg단위)", "수축기혈압", "이완기혈압", "식전혈당(공복혈당)", "총콜레스테롤", "트리글리세라이드", "HDL콜레스테롤", "LDL콜레스테롤", "(혈청지오티)AST"]
X[target] = scale.fit_transform(X[target])
x_test[target] = scale.transform(x_test[target])

# 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.25, random_state = 123)


# 모델링
model = Sequential()
model.add(Dense(15, input_dim = 25, activation = "relu")) # 입력층 - 15개, 은닉층 - 30개
model.add(Dense(1, activation = "sigmoid")) # 출력층 - 1개, sigmoid는 0, 1로 출력

# 모델 실행
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")
'''
"binary_crossentropy" - 이진분류
"adam" - 많이 쓰이는 최신형
metrics - 평가지표
'''
model.fit(X_tr, y_tr, epochs = 20, batch_size = 10)

# 모델 평가
y_pred = model.predict(X_val)
y_pred = tf.cast(y_pred >= 0.5, dtype=tf.int32)
print(accuracy_score(y_val, y_pred))

# 결과
model.fit(X, y, epochs = 20, batch_size = 10)
pred = model.predict(x_test)
pred = tf.cast(pred >= 0.5, dtype=tf.int32)
pred = pd.DataFrame(pred)[0]

result = pd.DataFrame({"ID":x_test_id, "흡연상태":pred})
result.to_csv("20230626.csv", index = False, encoding='utf-8-sig')

pd.read_csv("20230626.csv")


prediction = result.iloc[:,1]
y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/y_test.csv")
true = y_test.iloc[:,1]
print(accuracy_score(prediction, true)) # 정확도 60%대
