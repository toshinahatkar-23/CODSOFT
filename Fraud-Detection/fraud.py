import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data=pd.read_csv("creditcard.csv")
print(data.head())

X=data.drop('Class',axis=1)

y=data['Class']

print(X.head())
print(y.head())
print(y.value_counts())

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("Training size:",len(X_train))
print("Testing size:",len(X_test))


model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train,y_train)

print("\n--- Sample Prediction ---")
sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)
print("Prediction:", "Fraud" if pred[0]==1 else "Not Fraud")


predictions=model.predict(X_test)
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

print("\n--- Accuracy ---")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\n--- Classification Report ---")
print(classification_report(y_test,predictions))
