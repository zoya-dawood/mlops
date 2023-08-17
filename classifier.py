import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/mlops/data/Iris.csv")
features= ['SepalLengthCm',	'SepalWidthCm',	'PetalLengthCm','PetalWidthCm']
target = 'Species'
x_train,x_test,y_train,y_test =train_test_split(df[features],df[target], test_size=0.3, shuffle=True)
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(f"Accuracy of the model is {accuracy_score(y_test,y_pred)*100}")


