# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 01:30:25 2024

@author: 서다현
"""

'''
이용 데이터: 타이타닉 데이터
작업 유형: 생존자 분류
사용 모델: 텐서플로우 기반 인공신경망
'''

# 1. 패키지 임포트
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, roc_auc_score

# 2. 데이터 로드
data_files = [
    'USER_DIRECTORY/titanic/train.csv',
    'USER_DIRECTORY/titanic/test.csv'
]
train = pd.read_csv(data_files[0])
test = pd.read_csv(data_files[1])

# 3. 데이터 전처리
# (1) 범주형 변수 변환
train_test_data = [train, test]

for data in train_test_data :
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
mapping = {
    "Mr":0,"Miss":1,"Mrs":2,
    "Master":3, "Dr":3, "Rev":3, "Col": 3, 'Ms': 3, 'Mlle': 3, "Major": 3, 'Lady': 3, 'Capt': 3,
    'Sir': 3, 'Don': 3,'Dona': 3,'Mme':3, 'Jonkheer': 3, 'Countess': 3
}
for data in train_test_data :
    data['Title'] = data['Title'].map(mapping)

for data in train_test_data :
    if 'Name' in data.columns :
        data.drop('Name', axis=1, inplace=True)

ft = 'Sex'
train[ft].value_counts()
mapping = {'male':0, 'female':1}
for data in train_test_data :
    data[ft] = data[ft].map(mapping)
    
for data in train_test_data :
    data['Age'].fillna(data.groupby('Title')['Age'].transform('median'), inplace = True)
    
for data in train_test_data :
    cond1 = data['Age'] <= 16
    cond2 = (data['Age'] > 16) & (data['Age'] <= 26)
    cond3 = (data['Age'] > 26) & (data['Age'] <= 36)
    cond4 = (data['Age'] > 36) & (data['Age'] <= 46)
    cond5 = data['Age'] > 46

    data.loc[cond1, 'Age'] = 0
    data.loc[cond2, 'Age'] = 1
    data.loc[cond3, 'Age'] = 2
    data.loc[cond4, 'Age'] = 3
    data.loc[cond5, 'Age'] = 4
    
for data in train_test_data :
    data['Embarked'].fillna('S', inplace = True)
    
ft = 'Embarked'
mapping = {'S':0, 'Q':1, 'C':2}
for data in train_test_data :
    data[ft] = data[ft].map(mapping)
    
for data in train_test_data :
    data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'), inplace = True)
    
for data in train_test_data :
    cond1 = data['Fare'] <= 17
    cond2 = (data['Fare'] > 17) & (data['Fare'] <= 30)
    cond3 = (data['Fare'] > 30) & (data['Fare'] <= 100)
    cond4 = data['Fare'] > 100

    data.loc[cond1, 'Fare'] = 0
    data.loc[cond2, 'Fare'] = 1
    data.loc[cond3, 'Fare'] = 2
    data.loc[cond4, 'Fare'] = 3
    
for data in train_test_data :
    data.Cabin = data.Cabin.str[:1]
    
ft = 'Cabin'
mapping = {
    'C':0, 'E':0.1, 'G':0.2, 'D':0.3, 'A':0.4, 'B':0.5, 'F':0.6, 'T':0.7
} # 소수점으로 라벨링
for data in train_test_data :
    data[ft] = data[ft].map(mapping)
    
for data in train_test_data :
    data.Cabin.fillna(0.2, inplace = True)
    
for data in train_test_data :
    data['FamilySize'] = data.SibSp + data.Parch + 1
    
ft = 'FamilySize'
mapping = {
    1:0, 2:0.1, 3:0.2, 4:0.3, 5:0.4, 6:0.5, 7:0.6, 8:0.7, 11:0.8
} # 소수점으로 라벨링
for data in train_test_data :
    data[ft] = data[ft].map(mapping)
    
# (2) 사용 데이터 및 변수 선택
for data in train_test_data :
    data.drop(['SibSp', 'Parch', 'Ticket'], axis = 1, inplace = True)
    
X_train = train.drop(["Survived", "PassengerId"], axis = 1)
y_train = train["Survived"]

x_test_id = test.pop("PassengerId")

# 4. 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)

# 5. 모델링
model = Sequential()

model.add(Dense(15, input_dim = 8, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")
model.fit(X_tr, y_tr, epochs = 20, batch_size = 10)

# 6. 평가
y_pred = model.predict(X_val)
y_pred = tf.cast(y_pred >= 0.5, dtype = tf.int32)
print("accuracy score =", accuracy_score(y_val, y_pred), "roc auc score =", roc_auc_score(y_val, y_pred))

# 7. 적용
model.fit(X_train, y_train, epochs = 20, batch_size = 10)
pred = model.predict(test)
pred = tf.cast(pred >= 0.5, dtype=tf.int32)

predict = pd.DataFrame(pred)[0]
result = pd.DataFrame({"PassengerId":x_test_id, "Survived":predict})

result.to_csv("titanic_result.csv", index = False)