from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

model=LinearRegression()

# preparing data

numberOfHours=np.array([[1,2,3,4,5,6]])
marksGot=np.array([30,41,48,53,58,63])

#reshape X & initializing input output
X=numberOfHours.reshape(-1,1)
y=marksGot

model.fit(X,y)


numberOfHours=float(input("Enter number of hours you study: "))
hoursInput=np.array([[numberOfHours]])

prediction=model.predict(hoursInput)

print("You will get ",prediction[0]," marks ","if you study",hoursInput[0,0],"hours")

plt.plot(X,y, marker="o")
plt.title("Predicting Marks.")
plt.xlabel("Number Of Hours")
plt.ylabel("Marks Got")
plt.savefig("MarksGot.png", dpi=300)
plt.show()