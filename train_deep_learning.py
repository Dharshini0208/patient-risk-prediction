import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from tensorflow import keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
EarlyStopping = keras.callbacks.EarlyStopping


# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("Loading dataset...")

df = pd.read_csv(
    "dataset/patients.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================
# STEP 2: SELECT FEATURES
# ============================================

features = [
    "Age",
    "Gender",
    "BMI",
    "BloodPressure",
    "PreviousVisits",
    "MissedAppointments",
    "MedicationAdherence",
    "CarePlanUnderstanding",
    "FeedbackScore",
    "EngagementScore"
]

X = df[features]

y = df["RiskLevel"]


# ============================================
# STEP 3: ENCODE GENDER
# ============================================

X = pd.get_dummies(
    X,
    columns=["Gender"],
    dtype=int
)


# ============================================
# STEP 4: ENCODE TARGET
# ============================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

print("\nRisk classes:")
print(label_encoder.classes_)


# ============================================
# STEP 5: TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# STEP 6: FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================
# STEP 7: BUILD DEEP LEARNING MODEL
# ============================================

print("\nBuilding Neural Network...")

model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train_scaled.shape[1],)
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        3,
        activation="softmax"
    )
])


# ============================================
# STEP 8: COMPILE MODEL
# ============================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# STEP 9: EARLY STOPPING
# ============================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================
# STEP 10: TRAIN MODEL
# ============================================

print("\nTraining Deep Learning model...")

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================
# STEP 10B: TRAINING CURVES
# ============================================


# ============================================
# TRAINING VS VALIDATION ACCURACY
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Deep Learning Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "models/dl_accuracy_curve.png"
)

plt.show()


# ============================================
# TRAINING VS VALIDATION LOSS
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Deep Learning Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "models/dl_loss_curve.png"
)

plt.show()


print("\nTraining curves saved:")
print("models/dl_accuracy_curve.png")
print("models/dl_loss_curve.png")

print("\nDeep Learning training completed!")


# ============================================
# STEP 11: MAKE PREDICTIONS
# ============================================

y_proba = model.predict(
    X_test_scaled
)

y_pred = np.argmax(
    y_proba,
    axis=1
)


# ============================================
# STEP 12: MODEL EVALUATION
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

roc_auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="weighted"
)


# ============================================
# STEP 13: DISPLAY RESULTS
# ============================================

print("\n====================================")
print("DEEP LEARNING MODEL RESULTS")
print("====================================")

print(
    "Accuracy  :",
    round(accuracy, 4)
)

print(
    "Precision :",
    round(precision, 4)
)

print(
    "Recall    :",
    round(recall, 4)
)

print(
    "F1 Score  :",
    round(f1, 4)
)

print(
    "ROC-AUC   :",
    round(roc_auc, 4)
)


# ============================================
# STEP 14: CLASSIFICATION REPORT
# ============================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================
# STEP 15: CONFUSION MATRIX
# ============================================

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================
# STEP 16: SAVE MODEL
# ============================================

model.save(
    "models/patient_risk_deep_learning.keras"
)

joblib.dump(
    scaler,
    "models/dl_scaler.pkl"
)

joblib.dump(
    label_encoder,
    "models/dl_label_encoder.pkl"
)

joblib.dump(
    list(X.columns),
    "models/dl_features.pkl"
)


print("\n====================================")
print("DEEP LEARNING MODEL SAVED")
print("====================================")

print(
    "models/patient_risk_deep_learning.keras"
)

print(
    "models/dl_scaler.pkl"
)

print(
    "models/dl_label_encoder.pkl"
)

print(
    "models/dl_features.pkl"
)