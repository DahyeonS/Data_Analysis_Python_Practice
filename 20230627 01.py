# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:06:33 2023

@author: 서다현
"""

'''
이용 데이터: 고객 서비스 이탈 예측 데이터
작업 유형: 서비스 이탈 예측(분류)
사용 모델: 텐서플로우 기반 인공신경망
'''

# 패키지 임포트
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv")
x_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv")

# 데이터 전처리
X = x_train.drop("CustomerId", axis = 1)
y = y_train.drop("CustomerId", axis = 1)
x_test_id = x_test.pop("CustomerId")

label = LabelEncoder()

X["Surname"] = label.fit_transform(X["Surname"])
X["Geography"] = label.fit_transform(X["Geography"])
X["Gender"] = label.fit_transform(X["Gender"])
x_test["Surname"] = label.fit_transform(x_test["Surname"])
x_test["Geography"] = label.fit_transform(x_test["Geography"])
x_test["Gender"] = label.fit_transform(x_test["Gender"])

scale = MinMaxScaler()

target = ["CreditScore", "Balance", "EstimatedSalary"]
X[target] = scale.fit_transform(X[target])

# 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.3, random_state = 111)

# 모델링
model = Sequential()
model.add(Dense(20, input_dim = 11, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.summary()
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")

model.fit(X_tr, y_tr, epochs = 20, batch_size = 10)

# 모델 평가
y_pred = model.predict(X_val)
y_pred = tf.cast(y_pred >= 0.5, dtype = tf.int32)
print("accuracy_score =", accuracy_score(y_val, y_pred))

# 결과
model.fit(X, y, epochs = 20, batch_size = 10)
pred = tf.cast(model.predict(x_test) >= 0.5, dtype=tf.int32)
predict = pd.DataFrame(pred)[0]

result = pd.DataFrame({"CustomerId":x_test_id, "Exited":predict})

y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_test.csv")
print(accuracy_score(y_test.iloc[:,1], pred))

result.to_csv("20230627 01.csv", index = False)
pd.read_csv("20230627 01.csv")
