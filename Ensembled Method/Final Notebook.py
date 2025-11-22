import pandas as pd
from sklearn.metrics import roc_auc_score

# --- 1. Load Individual Predictions ---
# Load the test predictions from your previous submissions
try:
    # Prediction set 1 (Highest score)
    lgbm_sub = pd.read_csv('submission_lgbm_final_v2.csv') 
    # Prediction set 2 (Diverse score)
    xgb_sub = pd.read_csv('submission_xgb_enriched.csv') 
except FileNotFoundError:
    print("\n--- ERROR: COULD NOT FIND PREVIOUS SUBMISSION FILES ---")
    print("Please ensure 'submission_lgbm_final.csv' and 'submission_xgb_enriched.csv' are in your output directory.")


# --- 2. Define Weights Based on Validation Scores ---
# LGBM: 0.91920 (Higher Score)
# XGBoost: 0.91667 (Lower Score)

# Let's give the better LGBM model 60% of the weight and XGBoost 40%.
W_LGBM = 0.60
W_XGB = 0.40

# --- 3. Create the Weighted Ensemble Prediction ---
# We blend the 'loan_paid_back' column from both files.
final_predictions = (
    (W_LGBM * lgbm_sub['loan_paid_back']) + 
    (W_XGB * xgb_sub['loan_paid_back'])
)

# --- 4. Create Final Submission File ---
final_submission_df = pd.DataFrame({
    'id': lgbm_sub['id'], 
    'loan_paid_back': final_predictions
})

final_submission_df.to_csv('submission_final_ensemble.csv', index=False)
print("\nFINAL ENSEMBLE FILE CREATED!")
print(f"Blended Weights: LGBM ({W_LGBM:.2f}) + XGBoost ({W_XGB:.2f})")
