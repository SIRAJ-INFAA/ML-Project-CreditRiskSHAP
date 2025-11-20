# ML-Project-CreditRiskSHAP
Its Repo name

# Credit Risk Modeling with SHAP Interpretability
This project builds a credit risk classification model using XGBoost and interprets its predictions using SHAP (SHapley Additive exPlanations). It aims to provide transparent, explainable decisions for loan approvals by identifying key risk drivers and visualizing how each feature contributes to individual predictions.

## Core Concepts
- Credit Risk Modeling: Predicting whether a loan applicant is likely to repay or default.
- XGBoost Classifier: A gradient-boosted decision tree ensemble used for high-performance classification.
- SHAP (SHapley Additive Explanations): A game-theoretic approach to explain the output of machine learning models.
- Model Interpretability: Understanding how input features influence model predictions.
- RiskScore: A scaled feature used to rank applicants by predicted risk.

## Folder Structure
- `CreditRisk-ML.ipynb` – Main notebook with preprocessing, model training, SHAP analysis, and visualizations
- `requirements.txt `– Python dependencies
- `README.md` – Detailed project overview, insights, and interpretation
- `plots/` – SHAP and supporting visualizations
  - `shap_summary_plot.png`
  - `shap_case_high.png`
  - `shap_case_low.png`
  - `shap_case_borderline.png`
  - `riskscore_distribution.png`
  - `correlation_heatmap.png`
 - `data/` – Dataset used for training- synthetic_german_credit_spil.csv

## Installation
Install all required libraries using:

pip install -r requirements.txt

Dependencies:
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap

## Workflow Overview
1. Load and preprocess dataset
2. Train XGBoost model
3. Evaluate performance (AUC, Precision, Recall)
4. Generate global SHAP summary plot
5. Select 3 representative cases:
   - High-risk denial
   - Low-risk approval
   - Borderline case
6. Generate SHAP waterfall plots
7. Write textual analysis

## Workflow
-markdown
1. Data Preprocessing
- Label encoding for categorical features
- Standard scaling for numerical features (CreditAmount, Duration, Income, etc.)
2. Correlation Heatmap 
- A heatmap was generated to visualize feature relationships and detect multicollinearity.
- Strong positive correlation observed between CreditAmount and Duration.
3. Model Training & Optimization
- XGBoost classifier trained with GridSearchCV for hyperparameter tuning.
- Class imbalance addressed using scale_pos_weight.
- Best parameters selected based on AUC score.
4. Model Evaluation
- After tuning, the model achieved:
- AUC Score: 0.71 (improved from 0.526)
- Precision: 0.68
- Recall: 0.74
- Confusion Matrix: Balanced classification between approved and denied cases.
5. Global SHAP Analysis
- SHAP summary plot identifies top 5 risk drivers:
- CreditAmount
- Duration
- EmploymentSince
- CheckingAccount
- Age
6. RiskScore Distribution
- A histogram of RiskScore was plotted to justify the selection of borderline cases.
- Thresholds (0.2 < RiskScore < 0.5) were chosen based on the densest region of uncertainty.
7. Instance-Level SHAP Explanations ✅ (Expanded)
- Three representative cases were selected:
  -> High-risk denial
  -> Low-risk approval
  -> Borderline case
- SHAP waterfall plots were generated for each case and saved to the plot png/ folder. Each explanation maps SHAP values back to original feature meanings.

## Key Insights

- **Top 5 Risk Drivers**:
  1. CreditAmount
  2. Duration
  3. EmploymentSince
  4. CheckingAccount
  5. Age

- **SHAP Explanations**:
  - High-risk denial: High CreditAmount and long Duration pushed the model toward rejection.
  - Low-risk approval: Stable job and high income supported approval.
  - Borderline case: Mixed signals from CreditHistory and Housing.


## Detailed SHAP Interpretations
High-Risk Denial
- Prediction: Denied
- Key Drivers:
  - CreditAmount = 8,000 → High loan amount increases risk
  - Duration = 36 months → Longer repayment period adds uncertainty
  - CheckingAccount = 0 → No prior account history
- Interpretation: The model penalized the applicant for requesting a large loan over a long period without financial history.

Low-Risk Approval
- Prediction: Approved
- Key Drivers:
  - EmploymentSince = 4 years → Stable employment
  - Income = 5,000 → High income relative to loan
  - CreditHistory = 1 → Clean repayment record
- Interpretation: The model favored applicants with strong financial stability and repayment history.

Borderline Case
- Prediction: Approved (low confidence)
- Key Drivers:
  - CreditHistory = 0 → No credit history
  - Housing = 1 → Rented property
  - CreditAmount = 3,000 → Moderate loan size
- Interpretation: Mixed signals from moderate risk features placed this applicant near the decision boundary.

## Visuals
All plots are stored in the plot png/ folder:
- shap_summary_plot.png: Global feature importance.
- shap_case_high.png: SHAP explanation for high-risk denial.
- shap_case_low.png: SHAP explanation for low-risk approval.
- shap_case_borderline.png: SHAP explanation for borderline case.
- riskscore_distribution.png: Histogram used to justify borderline threshold.
- correlation_heatmap.png: Feature correlation matrix


## Lessons Learned
- Model performance improved significantly with proper tuning and class balancing.
- SHAP explanations provided actionable insights into model behavior.
- Visual justifications (heatmap, histogram) made the analysis more transparent and credible.
- Textual analysis was expanded to clearly link SHAP values to real-world financial reasoning.

## Author
Mohamed Siraj N
MCA Graduate | TNSKILLS
- Focused on AI/ML, Python, and model interpretability.
