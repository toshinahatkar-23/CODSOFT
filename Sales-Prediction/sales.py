import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data=pd.read_csv("advertising.csv")

print(data.head())

X=data[['TV','Radio','Newspaper']]

y=data['Sales']

print(X.head())
print(y.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("Training size:",len(X_train))
print("Testing size:",len(X_test))

model=LinearRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("MSE:",mean_squared_error(y_test,predictions))