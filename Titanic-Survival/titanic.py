import pandas as pd

data = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows:")
print(data.head())

data = data[['Survived', 'Pclass', 'Sex', 'Age']]

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

data['Age'] = data['Age'].fillna(data['Age'].mean())

data = data.dropna()

X = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()