import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
data = {
    "CGPA":[5,6,7,8,9,6.5,7.5,8.5],
    "Placed":[0,0,0,1,1,0,1,1]
}

df = pd.DataFrame(data)
X = df[["CGPA"]]
y = df["Placed"]

model = LogisticRegression()
model.fit(X,y)
cgpa = [[8]]
result = model.predict(cgpa)

if result[0] == 1:
    print("Student likely to be Placed")
else:
    print("Student likely to be Not Placed")
plt.scatter(df["CGPA"],df["Placed"])
plt.xlabel("CGPA")
plt.ylabel("Placement")
plt.title("Student Placement Prediction")
plt.show()
