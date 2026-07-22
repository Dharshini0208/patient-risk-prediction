import pandas as pd
import numpy as np

# Make random data reproducible
np.random.seed(42)

# Number of patients
n = 1000

# Create patient data
data = {
    "PatientID": [f"P{i:04d}" for i in range(1, n + 1)],

    "Age": np.random.randint(18, 80, n),

    "Gender": np.random.choice(
        ["Male", "Female"],
        n
    ),

    "BMI": np.round(
        np.random.uniform(18, 35, n),
        1
    ),

    "BloodPressure": np.random.randint(
        100, 180, n
    ),

    "PreviousVisits": np.random.randint(
        0, 15, n
    ),

    "MissedAppointments": np.random.randint(
        0, 6, n
    ),

    "MedicationAdherence": np.random.randint(
        30, 101, n
    ),

    "CarePlanUnderstanding": np.random.randint(
        1, 11, n
    ),

    "FeedbackScore": np.random.randint(
        1, 11, n
    ),

    "EngagementScore": np.random.randint(
        20, 101, n
    )
}

# Convert data into DataFrame
df = pd.DataFrame(data)


# Calculate patient risk score
risk_score = (
    df["MissedAppointments"] * 10
    + (100 - df["MedicationAdherence"]) * 0.3
    + (10 - df["CarePlanUnderstanding"]) * 5
    + (100 - df["EngagementScore"]) * 0.2
)


# Create risk level
# Create balanced risk categories
# Using quantiles gives approximately equal
# numbers of Low, Medium and High patients

df["RiskLevel"] = pd.qcut(
    risk_score,
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)


# Save dataset
df.to_csv(
    "dataset/patients.csv",
    index=False
)


# Display results
print("====================================")
print("DATASET CREATED SUCCESSFULLY")
print("====================================")

print("\nNumber of Patients:")
print(len(df))

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Patients:")
print(df.head())

print("\nRisk Level Distribution:")
print(df["RiskLevel"].value_counts())

print("\nDataset saved to:")
print("dataset/patients.csv")