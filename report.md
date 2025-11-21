# Interpretable Machine Learning for Credit Risk Modeling using SHAP Values

## Objective

The goal of this project was to build a transparent and interpretable credit risk classification model using XGBoost, with a strong focus on explainability through SHAP (SHapley Additive Explanations) values. Rather than optimizing purely for accuracy, the emphasis was on creating a model that could simulate real-world decision-making in regulated industries — where fairness, transparency, and accountability are essential.

## Model Selection

To tackle the classification task, I chose the XGBoost classifier due to its strong performance and native compatibility with SHAP. I used `GridSearchCV` to fine-tune key hyperparameters, including:

● `max_depth`
● `learning_rate`
● `n_estimators`
● `scale_pos_weight` (to address class imbalance)

This tuning process helped improve the model’s ability to generalize while ensuring that minority class predictions (i.e., defaults) were not overlooked.

## Performance Metrics

After tuning, the model showed a significant improvement over the baseline. The key evaluation metrics were:

● **AUC Score**: 0.71 (up from 0.526)
● **Precision**: 0.68
● **Recall**: 0.74

The confusion matrix indicated a balanced classification between approved and denied applicants, which is crucial for fairness in credit decisions.

## SHAP-Based Interpretability

### Global Feature Importance

Using SHAP’s summary plot, I identified the top five features that influenced the model’s predictions:

● **CreditAmount** – Higher loan amounts increased risk
● **Duration** – Longer repayment periods were associated with higher default probability
● **EmploymentSince** – Longer employment history reduced risk
● **CheckingAccount** – Lack of account history increased risk
● **Age** – Younger applicants showed slightly higher risk

### Instance-Level Explanations

To demonstrate local interpretability, I selected three representative cases:

● **High-Risk Denial**: The applicant requested a large loan over a long duration and had no checking account history. These factors pushed the model toward rejection.
● **Low-Risk Approval**: This applicant had a stable job, high income, and a clean credit history — all of which contributed positively to the approval decision.
● **Borderline Case**: The model showed uncertainty due to mixed signals — for example, no credit history but a moderate loan amount and stable housing. This case fell within the decision boundary and was ideal for manual review.

### Feature Mapping

To ensure transparency, I mapped SHAP values back to the original feature meanings. For instance:

● `CheckingAccount < 0` indicated no financial history
● `Duration > 36 months` suggested a longer repayment period, which increased risk

## RiskScore Distribution

To justify the selection of borderline cases, I plotted a histogram of the scaled `RiskScore`. The threshold range of 0.2 to 0.5 was chosen based on the densest region of uncertainty, where the model’s confidence was lowest. This helped identify applicants who deserved closer human review.

## Improvements Made

Throughout the project, I implemented several key improvements:

● Added a **correlation heatmap** to visualize feature relationships
● Expanded the **textual analysis** in both the notebook and README
● Tuned the model to significantly improve AUC and recall
● Provided **visual justifications** for borderline case selection
● Clearly linked SHAP values to **real-world financial reasoning**

## Conclusion

This project demonstrates how interpretable machine learning can be applied to credit risk modeling in a way that balances performance with transparency. By combining XGBoost with SHAP, I was able to build a model that not only predicts loan default risk effectively but also explains its decisions in a way that aligns with regulatory and ethical standards.

## Author

**Mohamed Siraj N**  
MCA Graduate | ASGARDIA Foundation – TNSKILLS  
Specialized in AI/ML, Python, and model interpretability.
