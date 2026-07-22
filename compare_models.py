import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# STEP 1: ENTER MODEL RESULTS
# ============================================

results = {
    "Model": [
        "Random Forest",
        "Deep Learning"
    ],

    "Accuracy": [
        0.8650,
        0.9350
    ],

    "Precision": [
        0.8667,
        0.9348
    ],

    "Recall": [
        0.8650,
        0.9350
    ],

    "F1 Score": [
        0.8654,
        0.9349
    ],

    "ROC-AUC": [
        0.9580,
        0.9947
    ]
}


# ============================================
# STEP 2: CREATE DATAFRAME
# ============================================

df = pd.DataFrame(results)


# ============================================
# STEP 3: DISPLAY MODEL COMPARISON
# ============================================

print("\n====================================")
print("MODEL PERFORMANCE COMPARISON")
print("====================================")

print(
    df.to_string(
        index=False
    )
)


# ============================================
# STEP 4: FIND BEST MODEL
# ============================================

best_model = df.loc[
    df["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = df[
    "Accuracy"
].max()


print("\n====================================")
print("BEST MODEL")
print("====================================")

print(
    "Best Model:",
    best_model
)

print(
    "Accuracy:",
    round(
        best_accuracy * 100,
        2
    ),
    "%"
)


# ============================================
# STEP 5: CREATE COMPARISON GRAPH
# ============================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]


rf_values = df[
    df["Model"] == "Random Forest"
][metrics].values[0]


dl_values = df[
    df["Model"] == "Deep Learning"
][metrics].values[0]


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    metrics,
    rf_values,
    marker="o",
    label="Random Forest"
)


plt.plot(
    metrics,
    dl_values,
    marker="o",
    label="Deep Learning"
)


plt.ylim(
    0.80,
    1.01
)


plt.title(
    "Random Forest vs Deep Learning"
)

plt.xlabel(
    "Evaluation Metrics"
)

plt.ylabel(
    "Score"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


# ============================================
# STEP 6: SAVE COMPARISON GRAPH
# ============================================

plt.savefig(
    "models/model_comparison.png"
)

plt.show()


print(
    "\nComparison graph saved to:"
)

print(
    "models/model_comparison.png"
)