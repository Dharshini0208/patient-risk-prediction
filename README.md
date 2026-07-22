## 1. Project Overview

The Patient Risk Prediction project is a Data Science and Artificial Intelligence-based system developed to predict the risk level of patients using healthcare-related data.

The system analyzes patient demographic, clinical, appointment, medication adherence, care plan understanding, feedback, and engagement information to classify patients into three risk categories:

- Low Risk
- Medium Risk
- High Risk

The project uses Machine Learning and Deep Learning techniques to build and evaluate predictive models. A Random Forest model is developed as the Machine Learning baseline, while a Neural Network is developed as the Deep Learning model. The performance of both models is compared using multiple evaluation metrics.

NLP-based patient feedback analysis is planned as the next stage of the project to extract additional insights from patient feedback.

---

## 2. Problem Statement

Healthcare organizations need effective methods to identify patients who may be at higher risk based on their health-related and engagement-related characteristics.

Manual identification of high-risk patients can be time-consuming and may not efficiently handle large amounts of patient data.

This project aims to develop a predictive system that analyzes patient information and classifies patients into Low, Medium, and High risk levels using Machine Learning and Deep Learning techniques.

---

## 3. Project Objectives

The main objectives of this project are:

1. To prepare and analyze a patient risk dataset.
2. To perform exploratory data analysis on patient information.
3. To preprocess and transform the input features.
4. To develop a Random Forest Machine Learning model.
5. To develop a Deep Learning neural network model.
6. To evaluate both models using multiple performance metrics.
7. To compare Machine Learning and Deep Learning model performance.
8. To identify important factors influencing patient risk prediction.
9. To select the best-performing model for future integration.
10. To analyze patient feedback using NLP in the next stage of development.

---

## 4. Dataset

The project uses a dataset containing 1000 patient records and 12 columns.

The main features used for prediction are:

- Age
- Gender
- BMI
- Blood Pressure
- Previous Visits
- Missed Appointments
- Medication Adherence
- Care Plan Understanding
- Feedback Score
- Engagement Score

The target variable is:

- RiskLevel

The RiskLevel contains three categories:

- Low
- Medium
- High

---

## 5. Data Preparation and Exploratory Data Analysis

The dataset was prepared and explored using Python and Pandas.

The following activities were performed:

- Dataset creation
- Dataset loading
- Dataset shape analysis
- Column identification
- Data type inspection
- Statistical summary
- Risk-level distribution analysis
- Feature selection
- Categorical feature encoding

The final dataset contains:

- 1000 patient records
- 12 columns

---

## 6. Machine Learning Model

A Random Forest classification model was developed as the baseline Machine Learning model.

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

The Random Forest model was trained to classify patients into Low, Medium, and High risk categories.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix

Feature importance analysis was also performed to identify the most influential features in patient risk prediction.

---

## 7. Deep Learning Model

A Deep Learning neural network was developed using TensorFlow and Keras.

The model includes:

- Dense neural network layers
- ReLU activation functions
- Dropout layers
- Softmax output layer
- Adam optimizer
- Sparse categorical cross-entropy loss
- Early stopping

Feature scaling was performed using StandardScaler before training the neural network.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix

Training and validation accuracy and loss curves were also generated to analyze model performance during training.

---

## 8. Model Performance Comparison

The performance of the Random Forest and Deep Learning models was compared.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 86.50% | 86.67% | 86.50% | 86.54% | 95.80% |
| Deep Learning | 93.50% | 93.48% | 93.50% | 93.49% | 99.47% |

Based on the evaluation results, the Deep Learning model achieved better overall performance than the Random Forest model.

Therefore, the Deep Learning model was selected as the best-performing predictive model for the current stage of the project.

---

## 9. Important Features

Feature importance analysis from the Random Forest model identified the following important factors:

1. Missed Appointments
2. Care Plan Understanding
3. Medication Adherence
4. Engagement Score
5. BMI
6. Age
7. Blood Pressure
8. Previous Visits
9. Feedback Score
10. Gender

These features provide useful information for understanding factors associated with patient risk classification.

---

## 10. Technologies Used

The following technologies and tools were used:

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Visual Studio Code
- GitHub

---

## 11. Project Structure

The project contains the following main files:

```text
patient-risk-prediction/
│
├── dataset/
│   └── patients.csv
│
├── models/
│   ├── patient_risk_model.pkl
│   ├── patient_risk_deep_learning.keras
│   ├── dl_scaler.pkl
│   ├── dl_label_encoder.pkl
│   ├── dl_features.pkl
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── dl_accuracy_curve.png
│   └── dl_loss_curve.png
│
├── src/
│   ├── prepare_dataset.py
│   ├── explore_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── train_deep_learning.py
│   └── compare_models.py
│
├── README.md
└── .gitignore
