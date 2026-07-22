# ============================================
# PATIENT RISK PREDICTION - MACHINE LEARNING
# ============================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("Loading dataset...")

df = pd.read_csv("dataset/patients.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================
# STEP 2: SEPARATE FEATURES AND TARGET
# ============================================

# PatientID is not useful for prediction
# So we remove it

X = df.drop(
    columns=["PatientID", "RiskLevel"]
)

# RiskLevel is what we want to predict

y = df["RiskLevel"]


print("\nFeatures used for prediction:")
print(X.columns.tolist())

print("\nTarget variable:")
print("RiskLevel")


# ============================================
# STEP 3: DEFINE CATEGORICAL COLUMNS
# ============================================

categorical_features = [
    "Gender"
]


# ============================================
# STEP 4: DEFINE PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================
# STEP 5: CREATE ML MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ============================================
# STEP 6: CREATE PIPELINE
# ============================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================
# STEP 7: SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# STEP 8: TRAIN MODEL
# ============================================

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================
# STEP 9: MAKE PREDICTIONS
# ============================================

y_pred = pipeline.predict(
    X_test
)


# ============================================
# STEP 10: EVALUATE MODEL
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================
# STEP 11: DISPLAY RESULTS
# ============================================

print("\n====================================")
print("MODEL EVALUATION RESULTS")
print("====================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================
# STEP 12: CLASSIFICATION REPORT
# ============================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================
# STEP 13: CONFUSION MATRIX
# ============================================

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================
# STEP 14: SAVE MODEL
# ============================================

joblib.dump(
    pipeline,
    "models/patient_risk_model.pkl"
)

print("\n====================================")
print("MODEL SAVED SUCCESSFULLY")
print("====================================")

print(
    "models/patient_risk_model.pkl"
)