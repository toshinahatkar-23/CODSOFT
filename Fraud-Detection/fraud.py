import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("creditcard.csv")
print(data.head())

X=data.drop('Class',axis=1)

y=data['Class']

print(X.head())
print(y.head())

print(y.value_counts())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("Training size:",len(X_train))
print("Testing size:",len(X_test))

model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

predictions=model.predict(X_test)

print(classification_report(y_test,predictions))