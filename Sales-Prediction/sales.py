import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data=pd.read_csv("advertising.csv")
print("\n--- Dataset Preview ---")
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

print("\n--- New Prediction ---")
new_data = pd.DataFrame([[200, 40, 50]],
columns=['TV','Radio','Newspaper'])

pred = model.predict(new_data)

print("Predicted Sales:", round(pred[0],2))

importance = pd.Series(model.coef_, index=X.columns)

print("\n--- Feature Importance ---")
print(importance)

print("\n--- Model Performance ---")
print("R2 Score:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))

plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
