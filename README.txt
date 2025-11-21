# ML_Project_CreditRiskSHAP  
**Repository Name:** `ML_Project_CreditRiskSHAP`

## Project Overview: Interpretable Machine Learning for Credit Risk Modeling using SHAP Values  
This project focuses on predicting loan default risk using a machine learning model (XGBoost) and explaining its decisions using SHAP values. Instead of just aiming for high accuracy, the goal is to make the model’s decisions understandable and fair — especially important in finance where transparency matters.

We used a synthetic credit dataset (similar to German Credit Data) and built a classifier that not only predicts risk but also explains why each applicant was approved or denied. This kind of interpretability is essential in regulated industries where black-box models are not acceptable.

## Model Used

 **XGBoost Classifier** – Chosen for its strong performance and compatibility with SHAP.
○**Hyperparameter Tuning** – Done using `GridSearchCV` to optimize:
  ● `max_depth`
  ● `learning_rate`
  ● `n_estimators`
  ● `scale_pos_weight` (to handle class imbalance)

## SHAP-Based Interpretability

● **Global SHAP Summary Plot** – Shows which features influence loan decisions the most.
● **Instance-Level SHAP Waterfall Plots** – Explain predictions for:
  ● A high-risk denial case
  ● A low-risk approval case
  ● A borderline case
 **Feature Mapping** – SHAP values are linked back to real-world meanings (e.g., `CheckingAccount < 0` means no financial history).

## Core Concepts

● **Credit Risk Modeling** – Predicting loan repayment likelihood.
● **XGBoost** – A powerful ensemble model using decision trees.
● **SHAP** – A method to explain model predictions using game theory.
● **Model Interpretability** – Understanding how features affect predictions.
● **RiskScore** – A scaled metric used to rank applicants by risk.


## How to Run the Project

```bash
git clone https://github.com/yourusername/credit-risk-shap-project.git
cd credit-risk-shap-project
```

## Folder Structure

```
credit-risk-shap-project/
│
├── CreditRisk-ML.ipynb         # Main notebook with all steps
├── requirements.txt            # Python dependencies
├── README.md                   # Project summary and insights
├── plots/                      # SHAP visualizations
│   ├── shap_summary_plot.png
│   ├── shap_case_high.png
│   ├── shap_case_low.png
│   ├── shap_case_borderline.png
│   ├── riskscore_distribution.png
│   └── correlation_heatmap.png
├── data/
│   └── synthetic_german_credit_spil.csv
```

## Installation

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_model.ipynb
```

**Dependencies:**  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`


## Workflow Summary

1. **Data Preprocessing**
   ● Label encoding for categorical features
   ● Scaling for numerical features (e.g., CreditAmount, Duration)

2. **Correlation Analysis**
   ● Heatmap to check feature relationships
   ● Found strong correlation between CreditAmount and Duration

3. **Model Training**
   ● Used XGBoost with GridSearchCV
   ● Balanced classes using `scale_pos_weight`

4. **Model Evaluation**
   ● AUC Score: **0.71** (improved from 0.526)
   ● Precision: **0.68**
   ● Recall: **0.74**
   ● Confusion Matrix: Balanced results

5. **Global SHAP Analysis**
   ● Top 5 risk drivers:
     ● CreditAmount
     ● Duration
     ● EmploymentSince
     ● CheckingAccount
     ● Age

6. **RiskScore Distribution**
   ● Histogram used to define borderline cases
   ● Threshold: **0.2 < RiskScore < 0.5**

7. **Instance-Level SHAP Explanations**
    ○ High-risk denial
    ○ Low-risk approval
    ○ Borderline case
   ● Each case explained using SHAP waterfall plots

## Key Insights

○ **Model Performance**
  ● AUC improved from 0.526 to 0.71
  ● Precision: 0.68, Recall: 0.74

○ **Top Risk Drivers**
  ● CreditAmount, Duration, EmploymentSince, CheckingAccount, Age

○ **Borderline Case Selection**
  ● Based on RiskScore histogram (0.2–0.5 range)

○ **Interpretability**
  ● SHAP clearly shows why each applicant was approved or denied

## SHAP Case Interpretations

### High-Risk Denial  
○ **Prediction:** Denied  
○ **Key Drivers:**
  ● CreditAmount = 8,000 → High loan amount
  ● Duration = 36 months → Long repayment period
  ● CheckingAccount = 0 → No financial history  
○ **Interpretation:** The model rejected the applicant due to high loan size and lack of financial background.

### Low-Risk Approval  
○ **Prediction:** Approved  
○ **Key Drivers:**
  ● EmploymentSince = 4 years → Stable job
  ● Income = 5,000 → Good income
  ● CreditHistory = 1 → Clean record  
○ **Interpretation:** Strong financial stability led to approval.

### Borderline Case  
○ **Prediction:** Approved (low confidence)  
○ **Key Drivers:**
  ● CreditHistory = 0 → No credit history
  ● Housing = 1 → Rented property
  ● CreditAmount = 3,000 → Moderate loan  
○ **Interpretation:** Mixed signals placed this applicant near the decision boundary.

## Visuals

All plots are saved in the `plots/` folder:

● `shap_summary_plot.png` – Global feature importance  
● `shap_case_high.png` – High-risk denial explanation  
● `shap_case_low.png` – Low-risk approval explanation  
● `shap_case_borderline.png` – Borderline case explanation  
● `riskscore_distribution.png` – Histogram for borderline threshold  
● `correlation_heatmap.png` – Feature correlation matrix

## Lessons Learned

● Tuning and class balancing improved model performance
● SHAP helped explain predictions clearly
● Visuals like heatmaps and histograms made the analysis more transparent
● Writing deeper analysis helped connect model logic to real-world reasoning

## Author

**Mohamed Siraj N**  
MCA Graduate | TNSKILLS  
Focused on AI/ML, Python, and making machine learning models more interpretable and useful.
