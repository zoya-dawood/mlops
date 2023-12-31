import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'a6632a9d-4bd9-4aa0-99ce-0d47bc549995'
resource_group = 'mlops-august'
workspace_name = 'intellipat-mlops'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='iris')
df=dataset.to_pandas_dataframe()

features= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
target = 'Species'
x_train,x_test,y_train,y_test =train_test_split(df[features],df[target], test_size=0.3, shuffle=True)
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(f"Accuracy of the model is {accuracy_score(y_test,y_pred)*100}")


