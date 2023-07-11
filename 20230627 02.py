# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:07:30 2023

@author: 서다현
"""

'''
이용 데이터: 이직 여부 판단 데이터
작업 유형: 이직 여부 예측(분류)
사용 모델: 텐서플로우 기반 인공신경망
'''

# 1. 패키지 로드
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, roc_auc_score

# 2. 데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_train.csv")
x_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_test.csv")

# 3. 데이터 전처리
# (1) 사용 데이터 선택
X = x_train.drop("enrollee_id", axis = 1)
y = y_train.drop("enrollee_id", axis = 1)
x_test_id = x_test.pop("enrollee_id")

# (2) EDA
X.info()
X.describe()

# (3) 결측값 처리
X.isna().sum()

null_tar = ["gender", "education_level"]
X[null_tar] = X[null_tar].fillna("Unknown")
x_test[null_tar] = x_test[null_tar].fillna("Unknown")

X["relevent_experience"] = X["relevent_experience"].fillna("No relevent experience")
X["enrolled_university"] = X["enrolled_university"].fillna("no_enrollment")
X["major_discipline"] = X["major_discipline"].fillna("No Major")
X["company_size"] = X["company_size"].fillna("50-99")
X["company_type"] = X["company_type"].fillna("Other")
X[["experience", "last_new_job"]] = X[["experience", "last_new_job"]].fillna(0)
x_test["relevent_experience"] = x_test["relevent_experience"].fillna("No relevent experience")
x_test["enrolled_university"] = x_test["enrolled_university"].fillna("no_enrollment")
x_test["major_discipline"] = x_test["major_discipline"].fillna("No Major")
x_test["company_size"] = x_test["company_size"].fillna("50-99")
x_test["company_type"] = x_test["company_type"].fillna("Other")
x_test[["experience", "last_new_job"]] = x_test[["experience", "last_new_job"]].fillna(0)

# (4) 범주형 변수 변환
X["experience"].unique()
X["experience"] = X["experience"].replace(">20", "21")
X["experience"] = X["experience"].replace("<1", "0")
X["experience"] = X["experience"].astype(int)
x_test["experience"] = x_test["experience"].replace(">20", "21")
x_test["experience"] = x_test["experience"].replace("<1", "0")
x_test["experience"] = x_test["experience"].astype(int)

X["last_new_job"].unique()
X["last_new_job"] = X["last_new_job"].replace(">4", "5")
X["last_new_job"] = X["last_new_job"].replace("never", "0")
X["last_new_job"] = X["last_new_job"].astype(int)
x_test["last_new_job"] = x_test["last_new_job"].replace(">4", "5")
x_test["last_new_job"] = x_test["last_new_job"].replace("never", "0")
x_test["last_new_job"] = x_test["last_new_job"].astype(int)

label = LabelEncoder()

X["city"] = label.fit_transform(X["city"])
X["gender"] = label.fit_transform(X["gender"])
X["relevent_experience"] = label.fit_transform(X["relevent_experience"])
X["enrolled_university"] = label.fit_transform(X["enrolled_university"])
X["education_level"] = label.fit_transform(X["education_level"])
X["major_discipline"] = label.fit_transform(X["major_discipline"])
X["company_size"] = label.fit_transform(X["company_size"])
X["company_type"] = label.fit_transform(X["company_type"])

x_test["city"] = label.fit_transform(x_test["city"])
x_test["gender"] = label.fit_transform(x_test["gender"])
x_test["relevent_experience"] = label.fit_transform(x_test["relevent_experience"])
x_test["enrolled_university"] = label.fit_transform(x_test["enrolled_university"])
x_test["education_level"] = label.fit_transform(x_test["education_level"])
x_test["major_discipline"] = label.fit_transform(x_test["major_discipline"])
x_test["company_size"] = label.fit_transform(x_test["company_size"])
x_test["company_type"] = label.fit_transform(x_test["company_type"])

# (5) 정규화
scale = MinMaxScaler()
X[["training_hours"]] = scale.fit_transform(X[["training_hours"]])

# 4. 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.3, random_state = 222)

# 5. 모델링
model = Sequential()

model.add(Dense(15, input_dim = 12, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")
model.fit(X_tr, y_tr, epochs = 20, batch_size = 10)

# 6. 모델 평가
y_pred = model.predict(X_val)
y_pred = tf.cast(y_pred >= 0.5, dtype = tf.int32)
print("accuracy score =", accuracy_score(y_val, y_pred), "roc auc score =", roc_auc_score(y_val, y_pred))

# 7. 결과
model.fit(X, y, epochs = 20, batch_size = 10)
pred = model.predict(x_test)
pred = tf.cast(pred >= 0.5, dtype=tf.int32)

y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_test.csv")
y_test = y_test.iloc[:,1]
print("accuracy score =", accuracy_score(y_test, pred), "roc auc score =", roc_auc_score(y_test, pred))

predict = pd.DataFrame(pred)[0]
result = pd.DataFrame({"enrollee_id":x_test_id, "target":predict})
result.to_csv("20230627 02.csv", index = False)

pd.read_csv("20230627 02.csv")
