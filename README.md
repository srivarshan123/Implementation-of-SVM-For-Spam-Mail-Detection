# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.start the program.
2. import pandas and read csv file.
3. print data head,info(),isnull().
4. assign x and y value as v1,v2.
5. import train_test_split
6. feature extraction import countvectorizer as cv
7. x_train cv transform fit.
8. import SVC and predict y and import metrics and find accuracy score.
9. stop the program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Srivarshan S 
RegisterNumber: 212221040163  
*/
import pandas as pd
data=pd.read_csv("/content/sample_data/spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
