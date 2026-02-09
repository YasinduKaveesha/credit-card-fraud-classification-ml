# Credit Card Fraud Detection using Machine Learning

**Latest Update:** Enhanced notebook suite with detailed decision tree analysis, complete SHAP explainability implementation, and refined ensemble evaluation (February 2026)

## Project Overview

This project addresses the critical problem of detecting fraudulent credit card transactions in highly imbalanced datasets. With fraud representing only ~0.17% of all transactions, the challenge is to maximize fraud detection (high recall) while maintaining low false positive rates. Traditional accuracy-focused metrics fail in this context; instead, we emphasize precision-recall trade-offs and evaluations using PR-AUC—the most informative metric for imbalanced classification.

## Dataset

The dataset contains credit card transactions with the following characteristics:

- **Source**: Credit Card Transactions dataset
- **Size**: ~284,807 transactions
- **Class Distribution**: ~0.17% fraudulent (492 fraud cases), 99.83% legitimate
- **Features**: 30 anonymized features derived from PCA transformation of original card transaction attributes
- **Preprocessing**: StandardScaler normalization applied; stratified train-test split (80/20) to preserve class distribution

Despite the extreme imbalance, stratified splitting ensures that modeling and evaluation metrics remain robust.

## Project Structure

The analysis progresses through four key modeling phases:

### 1. **03_dummy_and_logreg.ipynb** – Baseline Establishment
   - Dummy Classifier (most frequent and stratified strategies) as absolute baselines
   - Logistic Regression with `class_weight="balanced"` for interpretable linear baseline
   - Threshold analysis demonstrating precision-recall trade-offs
   - Establishes performance floor and confirms that simple models struggle with extreme imbalance

### 2. **04_decision_trees.ipynb** – Single Tree Models *(Enhanced)*
   - **Shallow Decision Tree** (max_depth=5): Controlled complexity, basic non-linear patterns
     - High recall but low precision (many false positives)
   - **Deep Decision Tree** (no max_depth): Unrestricted growth, high variance model
     - Improves precision but loses recall (overfitting demonstrated)
   - Demonstrates instability of single trees and motivates ensemble methods
   - Illustrates fundamental bias-variance trade-off under extreme class imbalance

### 3. **05_ensemble_models_rf_xgb.ipynb** – Ensemble Methods
   - **Random Forest**: Multiple decision trees with `class_weight="balanced"` and bag aggregation
   - **XGBoost**: Gradient boosting with `scale_pos_weight` (cl *(Complete Implementation)*
   - **Gain-Based Feature Importance**: Bar plot of top 20 features by contribution to splits
     - Identifies V14, V10, V12, V4 as primary fraud indicators
     - Non-uniform distribution confirms learned meaningful patterns
   - **SHAP Global Summary Plot**: Aggregates SHAP values across test set (2,000 sample subset)
     - Shows how each feature influences predictions in aggregate
     - Visualizes non-linear and asymmetric effects
   - **SHAP Waterfall (Local Explanations)**: Decomposes individual predictions
     - Base value + feature contributions → final fraud probability
     - Enables transparent explanations for specific transactions
   - Builds stakeholder trust through interpretable, defensible model decisions
   - SHAP global summary plots showing aggregate feature contributions
   - SHAP waterfall explanations for individual predictions
   - Builds trust and transparency for fraud detection deployment

## Modeling Approach

### Class Imbalance Handling

Fraud detection under extreme imbalance requires targeted strategies:

- **Dummy Classifier**: Establishes baseline (predicts majority class, fails to detect fraud)
- **Logistic Regression**: `class_weight="balanced"` penalizes false negatives more heavily
- **Decision Trees**: Shallow trees reduce overfitting; deep trees attempt to capture fraud patterns
- **Random Forest**: `class_weight="balanced"` combined with ensemble averaging for stability
- **XGBoost**: 
  - `scale_pos_weight = n_negative / n_positive` (≈580) increases penalty for fraud misclassification
  - `eval_metric="aucpr"` optimizes directly for PR-AUC rather than log loss
  - Captures non-linear relationships and feature interactions

### Training Configuration

- Train-test split: 80% / 20% (stratified)
- Feature scaling: StandardScaler (applied post-split to prevent leakage)
- Hyperparameter tuning: Grid and manual search for optimal model parameters
- No feature engineering beyond PCA-transformed features

## Evaluation Metrics

### Why These Metrics?

**Accuracy is deliberately avoided** because a model predicting all non-fraud achieves 99.83% accuracy while detecting zero fraud cases—useless in practice.

### Metrics Used

1. **PR-AUC (Precision-Recall Area Under Curve)**: Primary metric for imbalanced classification
   - Integrates precision and recall across thresholds
   - Unaffected by true negatives (majority class)
   - Most informative for fraud detection

2. **ROC-AUC**: Secondary metric for completeness
   - Useful for threshold-independent performance summary
   - Sensitive to both true positives and true negatives

3. **Precision & Recall**:
   - **Precision**: Percentage of flagged transactions that are actually fraudulent (minimizes false positives)
   - **Recall**: Percentage of actual fraud cases caught (minimizes false negatives)

4. **Confusion Matrix**: Detailed breakdown of true positives, false positives, false negatives, and true negatives to assess operational impact

## Final Results

**XGBoost is selected as the final model**, delivering the best balance of fraud detection and false positive control.

### XGBoost Performance (Test Set)

| Metric | Value |
|--------|-------|
| **PR-AUC** | 0.873 |
| **ROC-AUC** | 0.984 |
| **Precision** | 80.6% |
| **Recall** | 84.7% |
| **True Negatives** | 56,844 |
| **False Positives** | 20 |
| **True Positives** | 83 |
| **False Negatives** | 15 |

### Interpretation

- **Precision (80.6%)**: Of 103 transactions flagged as fraud, 83 are truly fraudulent, 20 are false alarms—acceptable in fraud context
- **Recall (84.7%)**: 83 out of 98 actual fraud cases detected; only 15 missed
- **False Positives**: Extremely low (20 out of 56,864 legitimate transactions), critical for customer experience
- **Superior to Random Forest**: Achieves +6.7% higher PR-AUC (0.873 vs 0.786) and +16% higher precision (80.6% vs 64.7%)

## Explainability

Transparency is essential for fraud detection systems deployed in regulated financial environments.

### Methods

1. **Gain-Based Feature Importance**: Identifies features that contribute most to splits in the XGBoost ensemble
2. **SHAP Global Summary**: Aggregates SHAP values across the test set to show which features drive fraud predictions
3. **SHAP Waterfall (Local)**: Explains individual predictions by decomposing the model output into feature contributions

### Insight

These techniques ensure stakeholders (risk teams, compliance, customers) understand *why* a transaction was flagged as suspicious, building trust in the automated system.

## Key Takeaways

- **Extreme Class Imbalance is Manageable**: With proper metrics (PR-AUC), class weighting, and ensemble methods, effective fraud detection is achievable even at 0.17% fraud rate
- **Ensemble Methods Outperform Linear Models**: Boosting and bagging capture non-linear fraud patterns better than logistic regression
- **Threshold Tuning Matters**: The decision threshold determines precision-recall trade-off; different business contexts may prefer higher precision (fewer customer complaints) or recall (fewer missed frauds)
- **Explainability Builds Confidence**: Feature importance and SHAP values demonstrate that model decisions are interpretable and defensible
- **Real-World Constraints Apply**: False positives create friction (legitimate customers blocked temporarily); false negatives result in financial loss—operational balance is key
- **Evaluation Matters Most**: Choosing appropriate metrics (PR-AUC > accuracy) is as important as choosing the right model

## Repository Organization

```
.
├── data/
│   ├── raw/                    # Original creditcard.csv
│   └── processed/v1_train_test/  # Scaled features, train/test splits, scaler joblib
├── models/
│   └── xgb_model.joblib        # Trained XGBoost model for deployment/explainability
├── notebooks/
│   ├── 03_dummy_and_logreg.ipynb      # Baselines and linear model
│   ├── 04_decision_trees.ipynb         # Tree-based single models
│   ├── 05_ensemble_models_rf_xgb.ipynb # Random Forest and XGBoost training
│   └── 06_explainability_xgb.ipynb     # SHAP and feature importance
├── results/
│   ├── figures/                # Visualizations and plots
│   ├── models/                 # Model artifacts (if any)
│   └── tables/                 # Results tables and metrics
└── README.md                   # This file
```
