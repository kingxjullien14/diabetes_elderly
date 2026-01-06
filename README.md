# Diabetes Prediction using BRFSS Data (2023 & 2024)
## Master's Project - CSP760

---

## Project Overview

This project implements a comprehensive diabetes prediction system using BRFSS (Behavioral Risk Factor Surveillance System) data from 2023 and 2024. The project fulfills the requirements of RO3 (Research Objective 3) by training machine learning models and implementing explainability methods (SHAP and LIME) for a diabetes risk assessment application.

## Project Structure

```
Dataset/
├── 2023_BRFSS_CLEANED.csv                    # 2023 BRFSS dataset
├── 2024_BRFSS_CLEANED.csv                    # 2024 BRFSS dataset
├── 1_data_preprocessing.ipynb                # Data loading and preprocessing
├── 2_exploratory_data_analysis.ipynb         # EDA and feature selection
├── 3_model_training_with_shap_lime.ipynb     # Model training with explainability
├── README.md                                 # This file
└── CSP750-RO3.pdf                           # Project requirements document
```

## Workflow

### Step 1: Data Preprocessing
**Notebook:** `1_data_preprocessing.ipynb`

**Objectives:**
- Load both 2023 and 2024 BRFSS datasets
- Combine datasets for comprehensive analysis
- Handle missing values using appropriate imputation strategies
- Remove duplicate records
- Detect and remove outliers using Isolation Forest
- Convert data types appropriately
- Create binary diabetes target variable
- Save preprocessed data for analysis

**Output:**
- `BRFSS_preprocessed.csv` - Clean, preprocessed dataset ready for analysis

### Step 2: Exploratory Data Analysis & Feature Selection
**Notebook:** `2_exploratory_data_analysis.ipynb`

**Objectives:**
- Analyze target variable distribution (class imbalance)
- Visualize univariate and bivariate relationships
- Perform correlation analysis
- Apply multiple feature selection methods:
  - Chi-Square test
  - ANOVA F-statistic
  - Mutual Information
  - Random Forest feature importance
  - Recursive Feature Elimination (RFE)
- Consolidate feature importance scores
- Select top features for modeling

**Outputs:**
- `selected_features.txt` - List of selected features
- `feature_importance_analysis.csv` - Detailed feature importance scores
- Comprehensive visualizations

### Step 3: Model Training & Explainability (RO3)
**Notebook:** `3_model_training_with_shap_lime.ipynb`

**Objectives:**
- Apply SMOTEENN to handle class imbalance
- Train three models:
  1. **Random Forest (RF)**
  2. **Support Vector Machine (SVM)**
  3. **XGBoost (XGB)**
- Evaluate models using multiple metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Matthews Correlation Coefficient
  - Confusion matrices
- Implement **SHAP (SHapley Additive exPlanations)**:
  - Global feature importance
  - Individual prediction explanations
  - Feature effect visualization
- Implement **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Local prediction explanations
  - Feature contribution analysis
- Generate explanation templates for app development
- Save best model and artifacts

**Outputs:**
- `best_diabetes_model.pkl` - Trained best model
- `feature_scaler.pkl` - Feature scaler for preprocessing
- `model_features.csv` - List of features used by the model
- `diabetes_risk_factors_ranking.csv` - Risk factors ranked by importance
- `model_comparison_results.csv` - Performance comparison of all models
- SHAP and LIME visualizations
- Explanation templates for app integration

## Key Technologies & Methods

### Machine Learning Models
1. **Random Forest** - Ensemble of decision trees
2. **Support Vector Machine (SVM)** - Kernel-based classifier
3. **XGBoost** - Gradient boosting framework

### Data Balancing
- **SMOTEENN** - Combines SMOTE (oversampling) and ENN (undersampling) to handle class imbalance

### Explainability Methods (RO3 Requirement)
1. **SHAP (SHapley Additive exPlanations)**
   - Provides global and local explanations
   - Shows feature contributions to predictions
   - Based on game theory (Shapley values)
   
2. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Explains individual predictions
   - Model-agnostic approach
   - Creates interpretable local models

### Feature Selection Methods
- Chi-Square Test
- ANOVA F-statistic
- Mutual Information
- Random Forest Importance
- Recursive Feature Elimination (RFE)

## Requirements

### Python Packages
```
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
imbalanced-learn
mlxtend
shap
lime
joblib
PyPDF2
```

Install all packages:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib imbalanced-learn mlxtend shap lime joblib PyPDF2
```

## How to Run

### Option 1: Run All Notebooks in Sequence
1. Open and run `1_data_preprocessing.ipynb`
2. Open and run `2_exploratory_data_analysis.ipynb`
3. Open and run `3_model_training_with_shap_lime.ipynb`

### Option 2: Run Individual Notebooks
Each notebook is self-contained but depends on outputs from previous notebooks.

## Expected Results

### Model Performance
Based on RO3 requirements and similar studies, expect:
- **Accuracy**: ~75-85%
- **ROC-AUC**: ~0.75-0.85
- **Sensitivity**: ~70-80% (important for catching high-risk cases)
- **Specificity**: ~75-85%

**Note:** XGBoost typically performs best, followed by Random Forest and SVM.

### Explainability Insights
SHAP and LIME analyses will reveal:
- **Top risk factors** (e.g., general health, BMI, age, blood pressure)
- **Feature interactions** and their effects on predictions
- **Individual explanations** showing why a specific person is classified as high/low risk

## Next Steps (RO3 Prototype Development)

1. **Prototype Implementation**
   - Use Streamlit or mobile app framework
   - Integrate trained model
   - Implement SHAP-based explanations in UI

2. **Usability Evaluation**
   - Recruit 5-8 older adults
   - Conduct think-aloud protocol testing
   - Administer System Usability Scale (SUS) questionnaire
   - Collect qualitative feedback

3. **Evaluation Metrics**
   - SUS score (target: >68)
   - Task completion rate
   - User trust and comprehension
   - Healthcare professional validation

## RO3 Compliance Checklist

- ✅ **Data Collection**: BRFSS 2023 & 2024 datasets
- ✅ **Data Preprocessing**: Cleaning, handling missing values, encoding, balancing
- ✅ **Model Training**: RF, SVM, XGBoost with cross-validation
- ✅ **Model Selection**: Best model identified and saved
- ✅ **Explainability (SHAP)**: Global and local explanations implemented
- ✅ **Explainability (LIME)**: Individual prediction explanations
- ✅ **Explanation Logic**: Templates and ranking for app development
- ✅ **Model Evaluation**: Comprehensive metrics (Accuracy, ROC-AUC, Sensitivity, Specificity, etc.)
- 🔲 **Prototype Implementation**: Next step (Streamlit/mobile app)
- 🔲 **Usability Testing**: Next step (5-8 older adults)
- 🔲 **Evaluation Report**: Next step (SUS, qualitative analysis)

## Key Files for App Development

When building the diabetes risk assessment app, you'll need:

1. **Model Files**
   - `best_diabetes_model.pkl` - The trained model
   - `feature_scaler.pkl` - For preprocessing user inputs
   - `model_features.csv` - List of required features

2. **Explanation Resources**
   - `diabetes_risk_factors_ranking.csv` - Feature importance ranking
   - Explanation templates from Notebook 3
   - SHAP analysis insights

3. **Example Code**
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('best_diabetes_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Load feature names
features = pd.read_csv('model_features.csv')['features'].tolist()

# Make prediction for new user
user_data = pd.DataFrame([user_input_dict])  # user_input_dict has all features
user_data_scaled = scaler.transform(user_data)
prediction = model.predict(user_data_scaled)[0]
probability = model.predict_proba(user_data_scaled)[0][1]

# Get SHAP explanation for this prediction
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_data_scaled)
# Use shap_values to show top contributing factors
```

## References

### Dataset
- BRFSS 2023 & 2024: Behavioral Risk Factor Surveillance System

### Methods
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP)
- Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System

### Evaluation
- Brooke, J. (1996). SUS: A "quick and dirty" usability scale
- Nielsen, J. (1994). Usability Engineering

## Troubleshooting

### Common Issues

1. **Memory Error during SMOTEENN**
   - Reduce training set size or use smaller sample
   - Increase system RAM or use cloud computing

2. **SHAP/LIME computation too slow**
   - Use smaller sample of test data (already implemented)
   - Reduce number of features
   - Use GPU acceleration if available

3. **Missing values in predictions**
   - Ensure all required features are present
   - Check feature names match exactly
   - Verify data types are correct

## Contact & Support

For questions about this implementation or the RO3 requirements, refer to:
- Course materials: CSP760
- Project specification: CSP750-RO3.pdf
- Your supervisor or course instructor

---

**Last Updated:** January 2026
**Author:** Masters Student - Part 3
**Course:** CSP760
