#------------------------------------------------------------
# Deep Learning PipeLine 
#------------------------------------------------------------
# 1. Read the data from CSV
# 2. Data Analysis (EDA)
# 3. PreProcessing
# 4. Train test Split
# 5. Feature Scaling
# 6. FNN Model Training 
# 7. Model Evaluation
# 8  Graphical Representation
# 9. Model Preserve 
# 10. Model Loading and preserve 
# 11. Test Unseen Data 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#------------------------------------------------------------
# 1. Read the data from CSV
#------------------------------------------------------------

print("1. Read the data from CSV")

data = pd.read_csv("placement_data.csv") 
print("Complete dataset")
print(data)

#------------------------------------------------------------
# 2. Data Analysis (EDA) 
#------------------------------------------------------------

print("2. Data Analysis (EDA)")

print("First 5 rows : ")
print(data.head())

print("Column names : ")
print(data.columns) 

print("Shape of dataset : ") 
print(data.shape)

print("Statistical Summary : ")
print(data.describe())

#------------------------------------------------------------
# 3. Pre-Processing
#------------------------------------------------------------

print("3. PreProcessing")

X = data[['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship']] 
Y = data["Placed"] 

print("Input Features : ") 
print(X.head()) 

print("Target : ")
print(Y.head()) 

#------------------------------------------------------------
# 4. Train test Split
#------------------------------------------------------------
print("4. Train test Split")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42 
) 

print("Training Input Shape : ", X_train.shape) 
print("Testing Input Shape : ", X_test.shape) 
print("Training Output Shape : ", Y_train.shape) 
print("Testing Output Shape : ", Y_test.shape)  

#------------------------------------------------------------
# 5. Feature Scaling
#------------------------------------------------------------

print("5. Feature Scaling")

scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test) 

print("Scaled training data : ")
print(X_train_scaled[:5]) 

#------------------------------------------------------------
# 6. FNN Model Training 
#------------------------------------------------------------
print("6. FNN Model Training ")

model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42 
)

print(model)
print("Train the model") 
model = model.fit(X_train_scaled, Y_train)
print("Model training completed") 

#------------------------------------------------------------
# 7. Model Evaluation
#------------------------------------------------------------

print("7. Model Evaluation")

Y_pred = model.predict(X_test_scaled) 
accuracy = accuracy_score(Y_test, Y_pred) 
print("Accuracy is : ", accuracy) 

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix : \n",cm)

print("Predict the probability : ")
Y_prob = model.predict_proba(X_test_scaled)
print(Y_prob[:5]) 

#------------------------------------------------------------
# 9. Model Preserve 
#------------------------------------------------------------

print("9. Model Preserve ")

joblib.dump(model, "placement_fnn_model.pkl")
joblib.dump(scaler, "placement_scaler.pkl") 

print("Model and scaler gets dumped successfully")

#------------------------------------------------------------
# 10. Model Loading and preserve 
#------------------------------------------------------------

print("10. Model Loading and preserve")

loaded_model = joblib.load("placement_fnn_model.pkl")
loaded_scaler = joblib.load("placement_scaler.pkl") 

print("Model gets loaded successfully") 

#------------------------------------------------------------
# 11. Test Unseen Data
# Aptitude :        70
# Coding :          75
# Communication :   80
# Academics :       85
# Internship :       1 
#------------------------------------------------------------

new_student = pd.DataFrame([[70, 75, 80, 85, 1]], columns=['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship'])  

new_student_scaled = loaded_scaler.transform(new_student) 

new_prediction = loaded_model.predict(new_student_scaled) 
new_probability = loaded_model.predict_proba(new_student_scaled) 

print("New Students data : ")
print(new_student) 

print("Prediction Probability : ", new_probability)

if new_prediction[0] == 1:
    print("Prediction : Placed")
else:
    print("Prediction : Not Placed") 