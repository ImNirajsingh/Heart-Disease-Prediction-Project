# ❤️ Heart Disease Prediction: An End-to-End Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green.svg)

## 📌 Project Overview
Cardiovascular diseases are the leading cause of death globally. Early detection and preventive care can significantly reduce mortality rates. This repository contains a complete, end-to-end Machine Learning solution designed to predict the likelihood of heart disease in patients based on easily obtainable health metrics and clinical parameters. 

The project was developed by Niraj at the Indian Institute of Technology, Guwahati. It bridges the gap between raw medical data and an accessible diagnostic tool by combining thorough Exploratory Data Analysis (EDA), rigorous data preprocessing, model training, and a fully interactive web application deployed via Streamlit.

## 📊 Dataset Description
The model is trained on a comprehensive heart disease dataset containing 918 patient records across 12 initial clinical attributes. The dataset includes a mix of categorical and numerical features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| **Age** | Numerical | Patient's age in years. |
| **Sex** | Categorical | Gender of the patient (M/F). |
| **ChestPainType** | Categorical | Type of chest pain: Typical Angina (TA), Atypical Angina (ATA), Non-Anginal Pain (NAP), or Asymptomatic (ASY). |
| **RestingBP** | Numerical | Resting blood pressure in mm Hg upon admission to the hospital. |
| **Cholesterol** | Numerical | Serum cholesterol measured in mg/dl. |
| **FastingBS** | Categorical | Fasting blood sugar. A value > 120 mg/dl indicates high blood sugar (1 = true; 0 = false). |
| **RestingECG** | Categorical | Resting electrocardiogram results (Normal, ST-T wave abnormality, or Left Ventricular Hypertrophy). |
| **MaxHR** | Numerical | Maximum heart rate achieved during physical exertion. |
| **ExerciseAngina** | Categorical | Exercise-induced angina (Y/N). |
| **Oldpeak** | Numerical | ST depression induced by exercise relative to a resting state. |
| **ST_Slope** | Categorical | The slope of the peak exercise ST segment (Up, Flat, Down). |
| **HeartDisease** | **Target** | **1 = High Risk of Heart Disease, 0 = Normal/Low Risk.** |

## 🛠️ Data Preprocessing & Exploratory Data Analysis (EDA)
To ensure the machine learning model learns genuine clinical patterns rather than noise, extensive data cleaning and preprocessing were performed in the `intro.ipynb` Jupyter Notebook[cite: 1]. 

### 1. Exploratory Data Analysis
* **Target Balance:** The dataset was checked for class imbalance, revealing 508 cases of heart disease and 410 normal cases[cite: 1].
* **Visualizations:** Count plots were generated using Seaborn to visualize the distribution of heart disease across categorical variables like `Sex`, `ChestPainType`, and `FastingBS`[cite: 1]. Histograms and KDE plots were used to understand the distributions of continuous variables like `Age` and `MaxHR`[cite: 1].

### 2. Handling Medically Impossible Values (Imputation)
During analysis, it was discovered that some patients had a `RestingBP` (Blood Pressure) and `Cholesterol` of `0`[cite: 1]. Because a living patient cannot have a blood pressure or cholesterol of zero, these were treated as missing data:
* **Cholesterol:** The `0` values were replaced with the mean of the non-zero cholesterol values (`244.64 mg/dL`)[cite: 1].
* **RestingBP:** The `0` values were replaced with the mean of the non-zero blood pressure values (`132.54 mm Hg`)[cite: 1].

### 3. Categorical Feature Encoding
Machine learning models require numerical inputs. Categorical variables were converted into binary columns using Pandas' `pd.get_dummies`[cite: 1]. The `drop_first=True` parameter was applied to drop the first category of each variable (e.g., dropping `Sex_F` and keeping `Sex_M`), effectively preventing the "dummy variable trap" (perfect multicollinearity) and expanding the dataset to 16 columns[cite: 1].

### 4. Feature Scaling
Because K-Nearest Neighbors (KNN) is a distance-based algorithm, features with larger scales (like Cholesterol in the 200s) would mathematically overpower features with smaller scales (like Oldpeak, which ranges from 0 to 6). To fix this, Scikit-Learn's `StandardScaler` was applied to standardise the numerical columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) so that they all contribute equally to the model's predictions[cite: 1].

## 🤖 Modeling Strategy
* **Data Splitting:** The preprocessed dataset was split into an 80% training set and a 20% testing set using a random state of 42 to ensure reproducibility[cite: 1].
* **Algorithm Selection:** While the Jupyter notebook includes exploratory setups with algorithms like `LinearRegression`[cite: 1], the final production model is a **K-Nearest Neighbors (KNN)** classifier (`KNN_heart.pkl`). KNN is highly effective for medical classification tasks as it diagnoses a new patient by finding the most historically similar patients in the dataset.
* **Artifact Serialization:** To ensure the web application perfectly mimics the training environment, the trained KNN model (`KNN_heart.pkl`), the fitted scaler (`scaler.pkl`), and the exact list of encoded columns (`columns.pkl`) were exported using `joblib`.

## 💻 Streamlit Web Application Workflow
The project features a sleek frontend built with **Streamlit**, allowing non-technical users (like doctors or patients) to interact with the model in real-time. 

**How the Backend Pipeline Works:**
1. **User Input:** The app captures patient details using intuitive UI components: sliders for continuous variables (e.g., Age, Max Heart Rate, Oldpeak), number inputs for precise measurements (Blood Pressure, Cholesterol), and dropdown menus for categorical data.
2. **Dictionary Mapping:** The inputs are mapped to a Python dictionary, dynamically appending prefixes to match the one-hot encoded format (e.g., converting a "Flat" slope input into `'ST_Slope_Flat': 1`)[cite: 2].
3. **Column Alignment:** The app converts this dictionary into a Pandas DataFrame. It then loops through the loaded `expected_columns` to ensure every required feature is present. If a dummy column is missing from the user's input, the app safely fills it with `0`[cite: 2].
4. **Scaling & Prediction:** The perfectly aligned DataFrame is passed through the pre-fitted `scaler`, and the scaled array is fed into the KNN model for inference[cite: 2]. 
5. **Output:** The app conditionally triggers a red error UI stating `"⚠ High Risk of Heart Disease"` if the model predicts a `1`, or a green success UI stating `"✅ Low Risk of Heart Disease"` if the model predicts a `0`[cite: 2].
<img width="1920" height="916" alt="Screenshot 2026-07-22 004525" src="https://github.com/user-attachments/assets/5f5e9873-ffbc-4f9f-ab17-7dae5bd5b6aa" />


<img width="1920" height="913" alt="Screenshot 2026-07-22 004548" src="https://github.com/user-attachments/assets/de66bbae-4c92-4a85-902b-c435caf3b830" />


<img width="1920" height="902" alt="Screenshot 2026-07-22 004742" src="https://github.com/user-attachments/assets/e413a7ed-a69f-4b48-89b0-dfb4e0408547" />


## 🚀 Installation & Local Setup Guide

Follow these steps to run the application on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/ImNirajsingh/Heart-Disease-Prediction-Project.git](https://github.com/ImNirajsingh/Heart-Disease-Prediction-Project.git)
cd heart-disease-prediction
