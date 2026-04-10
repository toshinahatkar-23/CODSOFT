from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


print("\n--- Dataset Preview ---")
data=load_iris()
print(data.data[:5])

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("Training size:",len(X_train))
print("Testing size:",len(X_test))

model=LogisticRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)

print("\n--- Sample Prediction ---")
sample = [[5.1, 3.5, 1.4, 0.2]]  # example flower
pred = model.predict(sample)
print("Predicted Class:", pred[0])
print("Flower Type:", data.target_names[pred[0]])

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

print("\n--- Model Performance ---")
print("Accuracy:",accuracy_score(y_test,predictions))

plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Classification")
plt.show()
