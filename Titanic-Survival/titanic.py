import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Titanic-Dataset.csv")
print(data.info())
print(data.describe())

print("First 5 rows:")
print(data.head())

data = data[['Survived', 'Pclass', 'Sex', 'Age']]
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'] = data['Age'].fillna(data['Age'].mean())
data = data.dropna()

X = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

passanger=pd.DataFrame([[3,0,22],
[1,1,35],
[2,0,38]],columns=['Pclass', 'Sex', 'Age'])
predictions=model.predict(passanger)

for i ,pred in enumerate(predictions):
    if pred==1:
       print(f"Passanger {i+1}: Survived")
    else:
       print(f"Passanger {i+1}:Did not Survive")



sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()
