# 🚀 Telecom Customer Churn Prediction Project 🚀

![Project Logo or Header Image](path-to-header-image)

## Overview

This project focuses on predicting customer churn for an Iranian telecom company using advanced machine learning techniques. By identifying key factors leading to customer churn, businesses can proactively address at-risk customers and enhance retention strategies.

We employ cutting-edge models such as **XGBoost** to handle imbalanced data, and interpret results using **SHAP** for explainability.

## 📊 Dataset

The dataset includes various features such as customer demographics, usage metrics (call durations, unique numbers called, etc.), and a churn label indicating whether the customer has left the service.

More details can be found at the [UCI Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset).

## 📈 Key Features

- **Data Preprocessing**: Handling missing values, scaling, and outlier removal.
- **Bivariate Analysis**: Age vs Churn and other factors to uncover key patterns.
- **VIF Values**: Checking for multicollinearity between independent variables.
- **Correlation Heatmap**: Visualizing relationships between features to identify strong correlations.
- **Modeling with XGBoost**: Hyperparameter-tuned gradient boosting model using RandomSearchCV.
- **Class Imbalance Handling**: Using XGBoost's inbuilt `scale_pos_weight` parameter.
- **Model Interpretability**: SHAP values used to explain feature importance and model behavior.
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC for model evaluation.

## 🔥 Why This Project Stands Out

1. **Real-World Application**: Offers a practical solution for businesses to minimize customer churn and retain valuable customers.
2. **Advanced Techniques**: XGBoost with RandomSearchCV ensures efficient and robust hyperparameter tuning.
3. **Explainable AI**: SHAP values make the model’s decisions transparent, critical for business decisions.
4. **Modular and Scalable**: Easily adaptable for other datasets and industries facing customer churn issues.

## 🔬 How It Works

1. **Data Preprocessing**: The dataset is cleaned, scaled, and preprocessed to handle missing data and outliers.
2. **Bivariate Analysis**: Age and churn are analyzed, and visualizations help uncover patterns. 
   
   ![Age vs Churn](path-to-age-vs-churn)
   
3. **Multicollinearity (VIF Analysis)**: Variance Inflation Factor (VIF) is calculated to check for multicollinearity in features.

   ![VIF Values](path-to-vif-values)
   
4. **Correlation Heatmap**: Helps identify strong correlations between features.

   ![Correlation Heatmap](path-to-correlation-heatmap)

5. **Model Training**: XGBoost is employed as the primary classifier, with RandomSearchCV used for hyperparameter optimization. 

6. **Churn Prediction**: Predictions are made on customer data to identify potential churners.

7. **Model Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix are computed for both GridSearchCV and RandomSearchCV models.

   - **GridSearchCV Model Performance:**
     - Accuracy: `X`
     - Precision: `X`
     - Recall: `X`
     - F1-Score: `X`
     - ROC-AUC: `X`

     ![ROC Curve - GridSearchCV](path-to-grid-roc-curve)
     ![Confusion Matrix - GridSearchCV](path-to-grid-confusion-matrix)

   - **RandomSearchCV Model Performance:**
     - Accuracy: `Y`
     - Precision: `Y`
     - Recall: `Y`
     - F1-Score: `Y`
     - ROC-AUC: `Y`

     ![ROC Curve - RandomSearchCV](path-to-random-roc-curve)
     ![Confusion Matrix - RandomSearchCV](path-to-random-confusion-matrix)

8. **Justification for RandomSearchCV**:
   - **Time Efficiency**: RandomSearchCV was much faster than GridSearchCV without compromising model performance.
   - **GridSearchCV vs RandomSearchCV**: 

     ```text
     GridSearchCV took X minutes while RandomSearchCV completed in Y minutes, showing a significant reduction in computation time.
     ```
     RandomSearchCV allows for more exploration of the parameter space, making it more efficient in practice for large datasets.

9. **Model Explainability with SHAP**: SHAP values are used to interpret individual predictions and feature importance.

   ![SHAP Summary Plot](path-to-shap-summary-plot)
   ![SHAP Dependence Plot](path-to-shap-dependence-plot)

10. **Saved Model Evaluation**: Evaluation of saved models on a sample of 100 customers for deployment readiness.
   
   - **Accuracy**: `Z`
   - **Precision**: `Z`
   - **Recall**: `Z`
   - **F1-Score**: `Z`
   - **ROC-AUC**: `Z`

## 📊 Results and Visualizations

### SHAP Summary Plot
This plot shows the impact of each feature on the model’s output.

![SHAP Summary Plot](path-to-shap-summary-plot)

### Feature Importance Plot
This plot highlights the most influential features in the churn prediction model.

![Feature Importance Plot](path-to-feature-importance-plot)

## 🛠️ Installation & Setup

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction

   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

4. **Explore the results**: Open the notebook to visualize model training, evaluation, and feature explanations.

## 📊 Model Evaluation

- **Accuracy**: How well the model predicts overall.
- **Precision & Recall**: Measuring the trade-off between false positives and false negatives.
- **F1-Score**: Balances precision and recall.
- **ROC-AUC**: Area under the curve, measuring model performance.

### Confusion Matrix
The confusion matrix shows the model’s performance in classifying churned vs. non-churned customers.

![Confusion Matrix](path-to-confusion-matrix)

## 🚀 Future Improvements

- **Advanced Models**: Experiment with deep learning architectures such as LSTM for sequential data.
- **Real-Time Prediction Pipeline**: Deploy the model as an API for real-time customer churn predictions.
- **Segmentation**: Explore customer segmentation techniques for more personalized retention strategies.

## 🤝 Contributions

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or features you'd like to add.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

## 👏 Acknowledgments

- Special thanks to the UCI Machine Learning Repository for the dataset.
- Libraries used: `XGBoost`, `SHAP`, `Scikit-Learn`, `Pandas`, `Matplotlib`.
