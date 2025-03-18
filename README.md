# Heart Disease Prediction System

## Overview

The **Heart Disease Prediction System** is a machine learning project that analyzes patient data to predict the likelihood of heart disease using classification techniques. It utilizes **Random Forest (RF)** and **Support Vector Classifier (SVC)** models as part of an ensemble learning approach to enhance prediction accuracy.

## Features

- Loads heart disease data from `heart.csv`
- Performs **data visualization** using Matplotlib and Seaborn
- Conducts **exploratory data analysis (EDA)**
- Implements **machine learning models (RF & SVC)**
- Uses **custom ensemble learning** for higher accuracy
- Predicts the likelihood of heart disease based on medical attributes

##  Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

2. Install Dependencies
Ensure you have Python installed (version 3.7+). Then install the required packages:

pip install numpy pandas matplotlib seaborn scikit-learn



3.Run the Script
Execute the Python script to perform data analysis:

python heartdisease.py

The dataset used is heart.csv, which includes the following attributes:
age	Age of the patient
sex	Gender (1 = Male, 0 = Female)
cp	Chest pain type
trestbps	Resting blood pressure (in mm Hg)
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar (> 120 mg/dl, 1 = True, 0 = False)
restecg	Resting electrocardiographic results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina (1 = Yes, 0 = No)
oldpeak	ST depression induced by exercise relative to rest
slope	Slope of the peak exercise ST segment
ca	Number of major vessels colored by fluoroscopy
thal	Thalassemia classification
target	1 = Heart disease present, 0 = No heart disease


The ensemble learning approach using Random Forest (RF) and Support Vector Classifier (SVC) resulted in an accuracy of 88.3%, outperforming traditional models.

Future Improvements
Implement deep learning models for better prediction.
Integrate a web-based user interface for easier interaction.
Enhance real-time prediction capabilities with an API.






