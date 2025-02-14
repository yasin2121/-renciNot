import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("student.csv")


print("İlk 3 satır:")
print(df.head(3), "\n")

print("Veri bilgisi:")
print(df.info(), "\n")


df.dropna(subset=["Marks", "number_courses", "time_study"], inplace=True)


X = df[["number_courses", "time_study"]]
y = df["Marks"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     
    random_state=42    
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)            
print(f"Test verisi üzerindeki MSE: {mse}")
print(f"Test verisi üzerindeki R^2: {r2}")


example_prediction = model.predict([[4, 10]])
print(f"\n4 ders ve 10 saat çalışma ile tahmin edilen not: {example_prediction[0]}")


max_marks = df["Marks"].max()
print(f"\nVeri kümesindeki maksimum not: {max_marks}")


train_score = model.score(X_train, y_train)
print(f"Eğitim verisi üzerindeki R^2: {train_score}")
