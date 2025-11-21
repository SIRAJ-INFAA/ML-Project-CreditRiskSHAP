# Interpretable Machine Learning for Credit Risk Modeling using SHAP Values
project title

## Objective

To build a transparent credit risk classification model using XGBoost and interpret its predictions using SHAP values. The goal is to simulate decision-making in regulated industries where fairness and explainability are critical.

##  Model Selection

- **XGBoost Classifier** was selected for its robustness and compatibility with SHAP.
- **GridSearchCV** was used to tune hyperparameters:
  - `max_depth`, `learning_rate`, `n_estimators`, `scale_pos_weight`
- **Class imbalance** was addressed using `scale_pos_weight`.

##  Performance Metrics

The model showed significant improvement over the baseline AUC of 0.526.

- AUC Score: 0.71  
- Precision: 0.68  
- Recall: 0.74  
- Confusion Matrix: Balanced classification between approved and denied cases

##  SHAP-Based Interpretability

### Global Feature Importance

SHAP summary plot revealed top contributors:
- CreditAmount
- Duration
- EmploymentSince
- CheckingAccount
- Age

### Instance-Level Explanations

- **High-Risk Denial**: Large loan amount, long duration, no account history
- **Low-Risk Approval**: Stable job, high income, clean credit history
- **Borderline Case**: Mixed signals from credit history and housing

### Feature Mapping

SHAP values were linked to original feature meanings:
- Example: `CheckingAccount < 0` → No financial history
- Example: `Duration > 36 months` → Longer repayment period increases risk

##  RiskScore Distribution

A histogram of scaled `RiskScore` was plotted to justify the selection of borderline cases. The threshold range (0.2–0.5) was chosen based on the densest region of uncertainty.

##  Improvements Made

- Added correlation heatmap to visualize feature relationships
- Expanded textual analysis in README and notebook
- Improved model performance through tuning
- Provided visual justification for borderline case selection
- Linked SHAP values to real-world financial reasoning

##  Conclusion

This project demonstrates how interpretable machine learning can be applied to credit risk modeling. By combining XGBoost with SHAP, we achieved both predictive performance and transparency — essential for real-world deployment in regulated domains.

##  Author

**Mohamed Siraj N**  
MCA Graduate | TNSKILLS  
- Specialized in AI/ML, Python, and model interpretability
