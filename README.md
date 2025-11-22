
-----

# 📑 Predicting Loan Payback: Tabular Playground Series

This repository documents the solution for the Kaggle Playground Series - Season 5, Episode 11 competition, focused on **predicting the probability a borrower will pay back their loan** (a binary classification task).

The project starts with a baseline model and systematically progresses through advanced ensemble techniques and data augmentation strategies to achieve a competitive final score.

## 🎯 Competition Goal

The primary objective was to predict the probability ($\text{P}(\text{loan\_paid\_back} = 1)$) for borrowers in the test set.

  * **Metric:** **Area Under the ROC Curve (ROC AUC)**.

## 🚀 Learning Journey and Technical Progression

This project was structured as a comprehensive learning exercise, progressing through three major stages of machine learning complexity:

| Stage | Model Used | Key Learning Concept | Initial Score | Final Score |
| :--- | :--- | :--- | :--- | :--- |
| **I. Baseline** | **Logistic Regression** | Linear Classification, ROC AUC Metric | $\approx 0.686$ | - |
| **II. Optimization** | **Random Forest** | Bagging Ensemble, Handling Non-Linearity | - | $\approx 0.905$ |
| **III. Advanced ML** | **LGBM & XGBoost** | Gradient Boosting Ensemble, Feature Engineering, Data Augmentation | - | **$\approx 0.919 - 0.923$** |

-----

## ⚙️ Methodology and Implementation Details

The final, high-scoring model uses a weighted ensemble of two robust Boosting algorithms trained on an enriched dataset.

### 1\. Data Augmentation (Enriched Training Data)

  * **Logic:** The competition dataset is synthetic. To improve model robustness, the training set was augmented by concatenating the competition data with the original, real-world **Loan Prediction Dataset** (as hinted in the competition description).
  * **Result:** The training data size increased from $\approx 594\text{k}$ rows to $\approx 614\text{k}$ rows, providing the models with more diverse patterns.

### 2\. Feature Engineering


Goal: Create features that highlight financial risk, overcoming the simplicity of the synthetic data's original columns.

Key Engineered Features:

Financial Ratios: Income_to_Loan_Ratio (annual_income / loan_amount)

Debt Metrics: Annual_Debt_Amount (debt_to_income_ratio * annual_income)

Payment Indicators: Payment_Burden (installment / monthly_income) (Utilizing the new columns from the enriched data).

### 3\. Final Ensemble

The ultimate submission was a **Weighted Average Ensemble** combining the predictions of two powerful models, leveraging their different error patterns:

  * **Model 1:** **LightGBM Classifier (LGBM)** (Generally faster and slightly higher score)
  * **Model 2:** **XGBoost Classifier (XGBoost)** (Provides model diversity and robust predictions)

| Model | Weight in Ensemble (Example) | OOF AUC |
| :--- | :--- | :--- |
| **LGBM** | $60\%$ | $\approx 0.91920$ |
| **XGBoost** | $40\%$ | $\approx 0.91667$ |

-----

## 💾 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `README.md` | This overview file. |
| `requirements.txt` | Lists all necessary Python packages (pandas, numpy, scikit-learn, lightgbm, xgboost). |
| `notebooks/` | Contains the Jupyter/Kaggle notebooks used for development. |
| `notebooks/LGBM_Final_Model.ipynb` | Code for the final **LGBM** model (Piece 1 of the ensemble) with Data Augmentation and Feature Engineering. |
| `notebooks/XGBoost_Ensemble_Piece.ipynb` | Code for the final **XGBoost** model (Piece 2 of the ensemble). |
| `submissions/` | Folder containing final submission files. |
| `submissions/submission_final_ensemble.csv` | **The final result file** (the blend of LGBM and XGBoost). |

-----

## 💻 How to Reproduce the Results

To run and verify the final ensemble, you must execute the two model notebooks and then run the ensemble script.

### Prerequisites

1.  **Python 3.x**
2.  Install libraries: `pip install -r requirements.txt` (see below for content).
3.  **Kaggle Setup:** You must have a Kaggle account and join the **Predicting Loan Payback** competition to access the data.

### `requirements.txt` Content

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
lightgbm>=4.0
xgboost>=2.0
```

### Steps

1.  **Download Data:** Download `train.csv`, `test.csv`, and the original dataset (`loan_dataset_20000.csv`) from Kaggle. Place them in a `/data` folder (or use Kaggle Notebooks).
2.  **Run LGBM:** Execute the `LGBM_Final_Model.ipynb` notebook to generate `submission_lgbm_final.csv`.
3.  **Run XGBoost:** Execute the `XGBoost_Ensemble_Piece.ipynb` notebook to generate `submission_xgb_enriched.csv`.
4.  **Run Ensemble:** Use the blend script (as provided in the final steps of the conversation) to load the two CSV files and create the final `submission_final_ensemble.csv`.

-----

## 💡 Key Takeaways

1.  **Model Diversity Matters:** Using XGBoost and LGBM together was more effective than fine-tuning a single model.
2.  **Data Quality:** Data Augmentation (using the original dataset) was essential for extracting the highest possible performance from the synthetic data.
3.  **Advanced Preprocessing:** Highly customized imputation and feature alignment are required when combining datasets with non-identical columns.
