import pandas as pd
data=pd.read_csv("movies.csv",encoding='latin1')
print(data.head())

data=data[['Year','Duration','Genre','Director','Actor 1','Actor 2','Actor 3','Rating']]



data['Year'] = data['Year'].astype(str)
data['Year'] = data['Year'].str.replace('-', '')
data['Year'] = data['Year'].str.replace('(', '')
data['Year'] = data['Year'].str.replace(')', '')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

data['Duration']=data['Duration'].astype(str).str.replace(' min','')
data['Duration']=pd.to_numeric(data['Duration'],errors='coerce')

data['Rating']=pd.to_numeric(data['Rating'],errors='coerce')


data=data.dropna(subset=['Rating'])

data['Year']=data['Year'].fillna(data['Year'].median())
data['Duration']=data['Duration'].fillna(data['Duration'].median())
data['Genre']=data['Genre'].fillna('unknown')
data['Director']=data['Director'].fillna('unknown')
data['Actor 1']=data['Actor 1'].fillna('unknown')
data['Actor 2']=data['Actor 2'].fillna('unknown')
data['Actor 3']=data['Actor 3'].fillna('unknown')


print("Cleaned Data:")
print(data.head())

from sklearn.preprocessing import LabelEncoder

le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor1 = LabelEncoder()
le_actor2 = LabelEncoder()
le_actor3 = LabelEncoder()


data['Genre']=le_genre.fit_transform(data['Genre'])
data['Director']=le_director.fit_transform(data['Director'])
data['Actor 1']=le_actor1.fit_transform(data['Actor 1'])
data['Actor 2']=le_actor2.fit_transform(data['Actor 2'])
data['Actor 3']=le_actor3.fit_transform(data['Actor 3'])

print("Encoded Data:")
print(data.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X=data[['Year','Duration','Genre','Director','Actor 1','Actor 2','Actor 3']]
Y=data['Rating']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=LinearRegression()
model.fit(X_train,Y_train)

predictions=model.predict(X_test)

print("MSE:",mean_squared_error(Y_test,predictions))
 