Credit Risk Modeling with SHAP Interpretability:
This project builds a credit risk classification model using XGBoost and interprets its predictions using SHAP (SHapley Additive exPlanations). It identifies key risk drivers and explains individual loan decisions to support transparent financial decision-making.

Folder Structure:
credit-risk-shap-project/
│
├── credit_risk_model.ipynb         # Main notebook with all steps
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview and insights
├── plots/                          # SHAP visualizations
│   ├── shap_summary_plot.png
│   ├── shap_case_high.png
│   ├── shap_case_low.png
│   ├── shap_case_borderline.png
│   └── riskscore_distribution.png


Required Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap



Workflow Overview:
Data Preprocessing
- Label encoding for categorical features
- Scaling of numerical features
Correlation Analysis
- Heatmap to visualize feature relationships
Model Training
- XGBoost classifier with hyperparameter tuning via GridSearchCV
- Class imbalance handled using scale_pos_weight
Model Evaluation
- Metrics: AUC, Precision, Recall, Confusion Matrix
Global SHAP Analysis
- Summary plot of top features influencing loan decisions
Instance-Level SHAP
- Waterfall plots for:
- High-risk denial
- Low-risk approval
- Borderline case
Textual Interpretation
- Detailed analysis of SHAP values and feature impact

Key Insights:
Model Performance
- AUC Score: Improved to 0.71 after tuning
- Precision: 0.68
- Recall: 0.74
- Confusion Matrix: Balanced classification
SHAP Feature Importance
Top 5 drivers of loan decisions:
- CreditAmount
- Duration
- EmploymentSince
- CheckingAccount
- Age
Instance-Level Explanations
- High-Risk Denial: High CreditAmount and long Duration pushed the model toward rejection.
- Low-Risk Approval: Stable Job, high Income, and long EmploymentSince supported approval.
- Borderline Case: Mixed signals from CreditHistory and Housing — ideal for manual review.

Visuals:
All plots are stored in the plots/ folder:
- shap_summary_plot.png: Global feature importance
- shap_case_high.png: High-risk denial explanation
- shap_case_low.png: Low-risk approval explanation
- shap_case_borderline.png: Borderline case explanation
- riskscore_distribution.png: Histogram used to select borderline threshold

👤 Author
Mohamed Siraj N
MCA Graduate | TNSKILLS
- Focused on AI/ML, Python, and model interpretability.