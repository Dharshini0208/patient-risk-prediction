import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.preprocessing import label_binarize


# ============================================
# STEP 1: LOAD DATASET
# ============================================

df = pd.read_csv(
    "dataset/patients.csv"
)


# ============================================
# STEP 2: LOAD TRAINED MODEL
# ============================================

model = joblib.load(
    "models/patient_risk_model.pkl"
)

print("Trained model loaded successfully!")


# ============================================
# STEP 3: PREPARE DATA
# ============================================

X = df.drop(
    columns=["PatientID", "RiskLevel"]
)

y = df["RiskLevel"]


# ============================================
# STEP 4: SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# STEP 5: PREDICTIONS
# ============================================

y_pred = model.predict(
    X_test
)

y_proba = model.predict_proba(
    X_test
)


# ============================================
# STEP 6: CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Patient Risk Prediction - Confusion Matrix"
)

plt.xlabel(
    "Predicted Risk Level"
)

plt.ylabel(
    "Actual Risk Level"
)

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png"
)

plt.show()


# ============================================
# STEP 7: ROC-AUC SCORE
# ============================================

# Get model class order
classes = model.classes_


# Calculate multi-class ROC-AUC
roc_auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="weighted"
)

print("\nWeighted ROC-AUC Score:")
print(
    round(roc_auc, 4)
)


# ============================================
# STEP 7B: ROC CURVE
# ============================================

# Convert class labels into binary format
y_test_binary = label_binarize(
    y_test,
    classes=classes
)


plt.figure(
    figsize=(8, 6)
)


# Plot ROC curve for each class
for i, class_name in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        y_proba[:, i]
    )

    class_auc = roc_auc_score(
        y_test_binary[:, i],
        y_proba[:, i]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {class_auc:.2f})"
    )


# Diagonal reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.title(
    "ROC Curve - Patient Risk Prediction"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "models/roc_curve.png"
)

plt.show()


# ============================================
# STEP 8: FEATURE IMPORTANCE
# ============================================

rf_model = model.named_steps[
    "model"
]

importance = rf_model.feature_importances_


feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()


feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")
print(
    feature_importance
)


# ============================================
# STEP 8B: FEATURE IMPORTANCE GRAPH
# ============================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "Feature Importance - Patient Risk Prediction"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Features"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png"
)

plt.show()


# ============================================
# FINAL OUTPUT
# ============================================

print(
    "\n===================================="
)

print(
    "EVALUATION COMPLETED"
)

print(
    "===================================="
)

print(
    "\nConfusion matrix saved to:"
)

print(
    "models/confusion_matrix.png"
)

print(
    "\nROC curve saved to:"
)

print(
    "models/roc_curve.png"
)

print(
    "\nFeature importance saved to:"
)

print(
    "models/feature_importance.png"
)