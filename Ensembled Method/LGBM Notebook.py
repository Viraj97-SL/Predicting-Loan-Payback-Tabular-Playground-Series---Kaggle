import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION (Set Paths and Column Names) ---
# NOTE: Using a single, fixed path is safer than combining path/filename.
COMP_PATH = "/kaggle/input/playground-series-s5e11/"
ORIG_FILE_PATH = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv" 

TARGET_COL = 'loan_paid_back'
print(f"Target Column: {TARGET_COL}")

# --- 1. Load and Prepare ALL Data ---
train_df = pd.read_csv(COMP_PATH + "train.csv")
test_df = pd.read_csv(COMP_PATH + "test.csv")

try:
    original_df = pd.read_csv(ORIG_FILE_PATH)
except FileNotFoundError:
    print("\n--- WARNING: ORIGINAL DATASET FILE NOT FOUND ---")
    print("Proceeding without original data.")
    original_df = None

test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# --- 2. ENRICHED TRAINING DATA (CONCATENATION) ---
if original_df is not None:
    print("Preparing Original Dataset for concatenation...")
    
    # 2a. Rename Original Columns
    original_df = original_df.rename(columns={
        'Loan_Status': TARGET_COL,
        'ApplicantIncome': 'annual_income', 
        'LoanAmount': 'loan_amount',
        'Credit_History': 'credit_score'
    })

    # 2b. Convert Original Target (Y/N) to Competition Target (1/0)

    if original_df[TARGET_COL].dtype == 'object':
         original_df[TARGET_COL] = original_df[TARGET_COL].map({'Y': 1, 'N': 0})
    original_df = original_df.dropna(subset=[TARGET_COL])

    # 2c. 🛑 FIX: The correct way to stack dataframes with different columns.
    # We use join='outer' to keep all columns from both dataframes.
    train_df = pd.concat([train_df, original_df], ignore_index=True, join='outer')
    print(f"Combined Training Data Size: {len(train_df)}")
    
    # 2d. 💡 CRITICAL FIX: Align Test Features (Add NaNs for new columns)
    # This prepares the test set for Feature Engineering and Imputation.
    new_features = [col for col in train_df.columns if col not in test_df.columns and col != TARGET_COL]
    for col in new_features:
        test_df[col] = np.nan
    print(f"Test data aligned with {len(new_features)} new features (filled with NaN).")


# --- 3. FEATURE ENGINEERING (FIXED and Simplified) ---
print("\nApplying Feature Engineering...")

# Feature 1: Income to Loan Ratio 
train_df['Income_to_Loan_Ratio'] = train_df['annual_income'] / train_df['loan_amount']
test_df['Income_to_Loan_Ratio'] = test_df['annual_income'] / test_df['loan_amount']

# Feature 2: Annual Debt Amount 
if 'debt_to_income_ratio' in train_df.columns:
    train_df['Annual_Debt_Amount'] = train_df['debt_to_income_ratio'] * train_df['annual_income']
    test_df['Annual_Debt_Amount'] = test_df['debt_to_income_ratio'] * test_df['annual_income']

# Feature 3: Monthly Payment Burden (Now works because new columns are aligned)
if 'installment' in train_df.columns:
    train_df['Payment_Burden'] = train_df['installment'] / train_df['monthly_income']
    test_df['Payment_Burden'] = test_df['installment'] / test_df['monthly_income']
    
print("Feature Engineering Complete.")


# --- 4. PREPROCESSING (Final Cleaning) ---
print("Starting Preprocessing...")
feature_cols = [col for col in train_df.columns if col != TARGET_COL]

# Imputation Loop 
for col in feature_cols:
    if train_df[col].dtype == 'object' or train_df[col].dtype == 'category':
        mode = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode)
        test_df[col] = test_df[col].fillna(mode)
    else:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

# Encoding and Alignment
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# Align columns to ensure both have the same one-hot encoded features
train_cols_to_keep = [col for col in train_df.columns if col != TARGET_COL]
test_df = test_df.reindex(columns=train_cols_to_keep, fill_value=0) 

# Split Data
X = train_df.drop(TARGET_COL, axis=1)
y = train_df[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 5. MODEL TRAINING (LGBM) ---
print("\nTraining the Final LGBM Model...")
lgbm_final_model = LGBMClassifier(n_estimators=200, learning_rate=0.03, random_state=42, n_jobs=-1)
lgbm_final_model.fit(X_train, y_train)

# Evaluate
val_preds_final = lgbm_final_model.predict_proba(X_val)[:, 1]
auc_score_final = roc_auc_score(y_val, val_preds_final)

print(f"FINAL Validation ROC AUC Score (Enriched Data): {auc_score_final}")


# --- 6. SUBMISSION ---
print("\nCreating Submission File...")
test_preds_final = lgbm_final_model.predict_proba(test_df)[:, 1]

submission_df_final = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_preds_final
})

# Save the file with a clear name
submission_df_final.to_csv('submission_lgbm_final_v2.csv', index=False)
