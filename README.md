# Diabetes Risk Assessment App for Older Adults
## Master's Project - CSP760

🔗 **Live Demo:** [Coming Soon - Deploy on Streamlit Cloud]

---

## Project Overview

This project implements a comprehensive diabetes prediction system using BRFSS (Behavioral Risk Factor Surveillance System) data from 2023 and 2024. The project fulfills the requirements of RO3 (Research Objective 3) by training machine learning models, implementing explainability methods (SHAP and LIME), and developing an elderly-friendly web application for diabetes risk assessment.

### Key Features
- 📊 **Machine Learning Models**: Random Forest, Logistic Regression, XGBoost, and LightGBM trained on BRFSS data
- 🧠 **AI Explainability**: SHAP and LIME integration for transparent predictions
- 👴 **Elderly-Friendly UI**: Large fonts, high contrast, progressive disclosure
- 🌐 **Bilingual Support**: English and Malay language options
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- ♿ **Accessibility**: WCAG AAA compliant for older adults
- 🔤 **Adjustable Text Size**: User-controlled font sizing (16-28px)
- ⌨️ **Keyboard Navigation**: Full keyboard shortcut support
- 🔊 **Screen Reader Support**: ARIA labels and semantic HTML

## Project Structure

```
Dataset/
├── Data Files
│   ├── 2023_BRFSS_CLEANED.csv                # 2023 BRFSS dataset
│   ├── 2024_BRFSS_CLEANED.csv                # 2024 BRFSS dataset
│   ├── BRFSS_preprocessed.csv                # Preprocessed combined dataset
│   ├── diabetes_risk_factors_ranking.csv     # Feature importance rankings
│   ├── feature_importance_analysis.csv       # Feature selection analysis
│   └── model_comparison_results.csv          # Model performance comparison
│
├── Notebooks (Research Workflow)
│   ├── 1_data_preprocessing.ipynb            # Data loading and preprocessing
│   ├── 2_exploratory_data_analysis.ipynb     # EDA and feature selection
│   └── 3_model_training_with_shap_lime.ipynb # Model training with explainability
│
├── Application Files
│   ├── diabetes_app_elderly.py               # 🎯 Main Streamlit application
│   ├── best_diabetes_model.pkl               # Trained LightGBM model
│   ├── feature_scaler.pkl                    # Feature scaler
│   ├── model_features.csv                    # Required features list
│   └── requirements.txt                      # Python dependencies
│
└── Documentation
    ├── README.md                             # This file
    ├── DESIGN_GUIDE.md                       # UI/UX design guidelines
    └── ELDERLY_UI_FEATURES.md                # Accessibility features
```

## Quick Start

### Run the Application Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/kingxjullien14/diabetes_elderly.git
   cd diabetes_elderly
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run diabetes_app_elderly.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Fork/Push this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select this repository and set:
   - Main file: `diabetes_app_elderly.py`
   - Python version: 3.10+
6. Click "Deploy"

---

## Research Workflow

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
- Train four models:
  1. **Random Forest (RF)**
  2. **Logistic Regression (LR)**
  3. **XGBoost (XGB)**
  4. **LightGBM (LGBM)**
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
1. **Random Forest (RF)** - Ensemble of decision trees
2. **Logistic Regression (LR)** - Linear classification model
3. **XGBoost (XGB)** - Gradient boosting framework
4. **LightGBM (LGBM)** - Efficient gradient boosting framework

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

### For Running the Application
```txt
streamlit==1.52.2
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.8.0
joblib==1.5.3
shap==0.50.0
matplotlib==3.10.8
```

### For Research Notebooks (Additional)
```
xgboost
seaborn
imbalanced-learn
mlxtend
lime
PyPDF2
```

Install all packages:
```bash
# For app only
pip install -r requirements.txt

# For full research environment
pip install -r requirements.txt xgboost seaborn imbalanced-learn mlxtend lime PyPDF2
```

## How to Run

### Option 1: Use the Streamlit App (Recommended for End Users)
```bash
streamlit run diabetes_app_elderly.py
```
This launches the interactive web application where users can:
- Answer health questions step-by-step
- Get diabetes risk predictions
- View personalized explanations (SHAP)
- Receive actionable health recommendations
- Switch between English and Malay

### Option 2: Run Research Notebooks (For Developers/Researchers)
1. Open and run `1_data_preprocessing.ipynb`
2. Open and run `2_exploratory_data_analysis.ipynb`
3. Open and run `3_model_training_with_shap_lime.ipynb`

Each notebook is self-contained but depends on outputs from previous notebooks.

## Application Features

### Elderly-Friendly Design
- **Large Fonts**: 18-24px minimum for readability
- **Adjustable Text Size**: User-controlled slider (16-28px range)
- **High Contrast**: WCAG AAA compliant (7:1 ratio)
- **Progressive Disclosure**: One question at a time to reduce cognitive load
- **Clear Navigation**: Large buttons (44x44px minimum touch targets)
- **Keyboard Shortcuts**: Full keyboard navigation support
  - `Alt + N`: Next step
  - `Alt + B`: Previous step
  - `Alt + H`: Toggle help
  - `Alt + R`: Restart assessment
  - `Tab`: Navigate between fields
- **Screen Reader Support**: ARIA labels, roles, and semantic HTML
- **Confirmation Dialogs**: Prevents accidental data loss
- **Help System**: Contextual tooltips and explanations
- **Dark Mode Support**: Automatic theme adaptation
- **Error Prevention**: Input validation and clear guidance

### User Journey
1. **Welcome Screen**: Simple introduction to the tool
2. **Step 1 - Personal Info**: Age, sex, education, employment
3. **Step 2 - Physical Health**: Weight, BMI, general health, checkup frequency
4. **Step 3 - Health Conditions**: Blood pressure and cholesterol medications, doctor visits
5. **Step 4 - Lifestyle**: Exercise habits and alcohol consumption
6. **Step 5 - Results**: 
   - Diabetes probability score
   - Risk level (Low/Moderate/High)
   - Top 5 contributing factors with SHAP explanations
   - Personalized action plan
   - Next steps recommendations

### Bilingual Support
- Full interface in English and Malay
- Easy language toggle
- Culturally appropriate messaging

---

## Expected Results

### Model Performance
Based on RO3 requirements and similar studies, expect:
- **Accuracy**: ~75-85%
- **ROC-AUC**: ~0.75-0.85
- **Sensitivity**: ~70-80% (important for catching high-risk cases)
- **Specificity**: ~75-85%

**Note:** LightGBM performed best in this implementation, followed by XGBoost, Random Forest, and Logistic Regression.

### Explainability Insights
SHAP and LIME analyses will reveal:
- **Top risk factors** (e.g., general health, BMI, age, blood pressure)
- **Feature interactions** and their effects on predictions
- **Individual explanations** showing why a specific person is classified as high/low risk

## Next Steps

### Completed ✅
- ✅ Data collection and preprocessing
- ✅ Model training and evaluation
- ✅ SHAP and LIME explainability implementation
- ✅ Streamlit prototype development
- ✅ Elderly-friendly UI design
- ✅ Bilingual support (EN/MS)
- ✅ Deployment-ready application

### In Progress 🚧
- 🚧 Deployment to Streamlit Cloud
- 🚧 Usability testing with older adults (5-8 participants)
- 🚧 System Usability Scale (SUS) evaluation

### Future Enhancements 🔮
- Mobile app version (React Native/Flutter)
- Integration with healthcare provider systems
- Additional language support
- Voice input/text-to-speech for accessibility
- PDF report generation
- Audio feedback and cues

## RO3 Compliance Checklist

### Data & Model Development
- ✅ **Data Collection**: BRFSS 2023 & 2024 datasets
- ✅ **Data Preprocessing**: Cleaning, handling missing values, encoding, balancing
- ✅ **Model Training**: RF, Logistic Regression, XGBoost, LightGBM with cross-validation
- ✅ **Model Selection**: LightGBM selected as best model (saved)
- ✅ **Explainability (SHAP)**: Global and local explanations implemented
- ✅ **Explainability (LIME)**: Individual prediction explanations
- ✅ **Model Evaluation**: Comprehensive metrics (Accuracy, ROC-AUC, Sensitivity, Specificity)

### Prototype Development
- ✅ **Application Framework**: Streamlit web application
- ✅ **User Interface**: Elderly-friendly design (large fonts, high contrast, progressive disclosure)
- ✅ **Accessibility**: WCAG AAA compliance for older adults
- ✅ **Adjustable Text Size**: User-controlled font sizing (16-28px slider)
- ✅ **Keyboard Navigation**: Alt+N/B/H/R shortcuts for all navigation
- ✅ **Screen Reader Support**: ARIA labels, roles, and sr-only text
- ✅ **Confirmation Dialogs**: Prevents accidental actions (e.g., restart)
- ✅ **Multilingual**: English and Malay support
- ✅ **SHAP Integration**: Top 5 factors displayed with explanations
- ✅ **Personalized Recommendations**: Actionable health advice based on predictions
- ✅ **Deployment Ready**: requirements.txt and model files included

### Evaluation (Next Steps)
- 🔲 **Usability Testing**: 5-8 older adults with think-aloud protocol
- 🔲 **SUS Questionnaire**: System Usability Scale (target: >68)
- 🔲 **Qualitative Analysis**: User feedback and trust assessment
- 🔲 **Healthcare Validation**: Professional review of recommendations
- 🔲 **Evaluation Report**: Comprehensive findings and improvements

## Key Files for App Development

The Streamlit app (`diabetes_app_elderly.py`) uses these essential files:

1. **Model Files** (Required for predictions)
   - `best_diabetes_model.pkl` - Trained LightGBM model (best performing)
   - `feature_scaler.pkl` - StandardScaler for preprocessing user inputs
   - `model_features.csv` - List of 19 required features in correct order

2. **Application Code**
   - `diabetes_app_elderly.py` - Main Streamlit application (~1600 lines)
   - Includes bilingual translations (English/Malay)
   - SHAP explanations for top 5 risk factors
   - Personalized recommendations engine
   - Keyboard shortcuts and accessibility features

3. **Dependencies**
   - `requirements.txt` - Python package versions for deployment

### How the App Works

```python
# Simplified workflow
import joblib
import shap

# 1. Load model and preprocessing tools
model = joblib.load('best_diabetes_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
features = pd.read_csv('model_features.csv')['features'].tolist()

# 2. Collect user input through 5-step form
user_data = collect_user_responses()  # Returns dict with all features

# 3. Preprocess and predict
user_df = pd.DataFrame([user_data])
user_scaled = scaler.transform(user_df)
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

# 4. Generate SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_scaled)

# 5. Display results with explanations
- Probability score (0-100%)
- Risk level (Low/Moderate/High)
- Top 5 contributing factors
- Personalized action plan
```

## References

### Dataset
- **CDC** (2023, 2024). Behavioral Risk Factor Surveillance System (BRFSS). Centers for Disease Control and Prevention.

### Machine Learning & Explainability
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.

### UI/UX Design for Older Adults
- Amouzadeh, E., Dianat, I., Faradmal, J., & Babamiri, M. (2025). Optimizing Mobile App Design for Older Adults: Systematic Review of Age-Friendly Design. *Aging Clinical and Experimental Research*, 37, 248. https://doi.org/10.1007/s40520-025-03157-7
- Arch, A. (2008). Web Accessibility for Older Users: A Literature Review. W3C Working Draft.
- Czaja, S. J., & Lee, C. C. (2007). The Impact of Aging on Access to Technology. *Universal Access in the Information Society*, 5(4), 341-349.
- Fisk, A. D., Rogers, W. A., Charness, N., Czaja, S. J., & Sharit, J. (2009). *Designing for Older Adults: Principles and Creative Human Factors Approaches*. CRC Press.
- Gomez-Hernandez, M., Ferre, X., Moral, C., & Villalba-Mora, E. (2023). Design Guidelines of Mobile Apps for Older Adults: Systematic Review and Thematic Analysis. *JMIR mHealth and uHealth*, 11, e43186. https://doi.org/10.2196/43186
- Hawthorn, D. (2000). Possible Implications of Aging for Interface Designers. *Interacting with Computers*, 12(5), 507-528.
- He, H., Raja Ghazilla, R. A., & Abdul-Rashid, S. H. (2025). A Systematic Review of the Usability of Telemedicine Interface Design for Older Adults. *Applied Sciences*, 15(10), 5458. https://doi.org/10.3390/app15105458
- Johnson, J., & Finn, K. (2015). *Designing User Interfaces for an Aging Population: Towards Universal Design*. Morgan Kaufmann.
- Lim, K. H., Lim, C. Y., Achuthan, A., Wong, C. E., & Tan, V. P. S. (2024). The Review of Malaysia Digital Health Service Mobile Applications' Usability Design. *International Journal of Advanced Computer Science and Applications*, 15(10), 139-148.
- Liu, Z., & Yu, X. (2024). Development of a T2D App for Elderly Users: Participatory Design Study via Heuristic Evaluation and Usability Testing. *Electronics*, 13(19), 3862. https://doi.org/10.3390/electronics13193862
- Nielsen, J. (1994). *Usability Engineering*. Morgan Kaufmann Publishers.
- Nielsen, J. (2013). Usability for Senior Citizens: Improved, But Still Lacking. Nielsen Norman Group.
- W3C Web Accessibility Initiative (2018). Web Content Accessibility Guidelines (WCAG) 2.1. World Wide Web Consortium.

### Usability Evaluation
- Brooke, J. (1996). SUS: A "Quick and Dirty" Usability Scale. In P. W. Jordan, B. Thomas, B. A. Weerdmeester, & I. L. McClelland (Eds.), *Usability Evaluation in Industry* (pp. 189-194). Taylor & Francis.

## Troubleshooting

### Application Issues

1. **App won't start**
   ```bash
   # Check if all dependencies are installed
   pip install -r requirements.txt
   
   # Verify model files exist
   ls best_diabetes_model.pkl feature_scaler.pkl model_features.csv
   ```

2. **Model files not found**
   - Ensure you're running from the correct directory
   - Model files must be in the same directory as `diabetes_app_elderly.py`

3. **SHAP explanations slow**
   - Normal for first run (SHAP builds explainer)
   - Subsequent predictions are faster

### Notebook Issues

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

## Contributing

This is a research project for CSP760. For questions or collaborations:
- Open an issue on GitHub: https://github.com/kingxjullien14/diabetes_elderly/issues
- Contact the project supervisor

## License

This project is developed for academic purposes as part of a Master's degree program.

---

**Repository:** https://github.com/kingxjullien14/diabetes_elderly  
**Last Updated:** January 2026  
**Author:** Masters Student - Part 3  
**Course:** CSP760
