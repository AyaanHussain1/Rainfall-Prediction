import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle

final_sample = pd.read_csv("final_data.xls")

x = final_sample.drop(columns=["rainfall"])
y = final_sample["rainfall"]
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)

model = RandomForestClassifier()
param_grid = {
    "n_estimators" : [50,100,200],
    "max_features" : ["sqrt","log2"],
    "max_depth" : [None,10,20,30],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]}

grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,n_jobs=-1,verbose=0)
grid_search.fit(x_train,y_train)
pred = grid_search.predict(x_train)
a = accuracy_score(y_train,pred)
print(f"Train Score : {a}")

best_model = grid_search.best_estimator_

pred_test = best_model.predict(x_test)
b = accuracy_score(y_test,pred_test)
print(f"Test Score : {b}")

cv_scores = cross_val_score(best_model,x_train,y_train,cv=5)
print("Cross Validation Score: ",cv_scores)
print("Mean Cross Validation Score: ",np.mean(cv_scores))
print("Classification Report: ",classification_report(y_test,pred_test))

model_data = {"model":best_model,"feature_names":x.columns.tolist()}
with open("model.pkl","wb") as file:
    pickle.dump(model_data,file)

model = model_data["model"]
feature_names = model_data["feature_names"]

input_data = (1015.9,19.9,95,81,0.0,40.,13.7)
input_df = pd.DataFrame([input_data],columns = feature_names)

prediction = best_model.predict(input_df)
if prediction[0] == 1:
    print("Rainfall")
else:
    print("No Rainfall")