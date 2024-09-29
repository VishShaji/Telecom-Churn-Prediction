# 🚀 Telecom Customer Churn Prediction Project 🚀

![Project Logo](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/Customer_Churn_Prediction_Models.png)

## Overview

This project focuses on predicting customer churn for an Iranian telecom company using advanced machine learning techniques. By identifying key factors leading to customer churn, businesses can proactively address at-risk customers and enhance retention strategies.

We employ cutting-edge models such as **XGBoost** to handle imbalanced data, and interpret results using **SHAP** for explainability.

## 📊 Dataset

The dataset includes various features such as customer demographics, usage metrics (call durations, unique numbers called, etc.), and a churn label indicating whether the customer has left the service.

![Data head](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/data_head.png)

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
2. **Univariate Analysis**: Each variable is analyzed individually to visualize the distribution.
3. **Bivariate Analysis**: Age and churn are analyzed against other features, and visualized to help uncover patterns. 
   
   ![Age](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/age_bivar.png))
   ![Churn](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/churn_bivar.png)

   
4. **Multicollinearity (VIF Analysis)**: Variance Inflation Factor (VIF) is calculated to check for multicollinearity in features.

   ![VIF Values](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/vif.png)
   
5. **Correlation Heatmap**: Helps identify strong correlations between features.

   ![Correlation Heatmap](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/corr.png)

6. **Model Training**: XGBoost is employed as the primary classifier, with RandomSearchCV used for hyperparameter optimization. 

7. **Churn Prediction**: Predictions are made on customer data to identify potential churners.

8. **Model Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix are computed for both GridSearchCV and RandomSearchCV models.

   **GridSearchCV Model Performance:**
   - **Accuracy**: `0.958`
   - **Precision**: `0.802`
   - **Recall**: `0.931`
   - **F1-Score**: `0.861`
   - **ROC-AUC**: `0.9868`

     ![ROC Curve - GridSearchCV](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/roc_grid.png)
     ![Confusion Matrix - GridSearchCV](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/confusion_grid.png)

   **RandomSearchCV Model Performance:**
   - **Accuracy**: `0.958`
   - **Precision**: `0.796`
   - **Recall**: `0.942`
   - **F1-Score**: `0.863`
   - **ROC-AUC**: `0.9848`

     ![ROC Curve - RandomSearchCV](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/roc_random.png)
     ![Confusion Matrix - RandomSearchCV](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/confusion_random.png)

9. **Justification for RandomSearchCV**:
   - **Time Efficiency**: RandomSearchCV was much faster than GridSearchCV without compromising model performance.
   - **GridSearchCV vs RandomSearchCV**: 

     ![Text_GridTime_vs_RandomTime](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/grid_vs_random.png)
 
     RandomSearchCV allows for more exploration of the parameter space, making it more efficient in practice for large datasets.

10. **Model Explainability with SHAP**: SHAP values are used to interpret individual predictions and feature importance.

11. **Saved Model Evaluation**: Evaluation of saved models on a sample of 100 customers for deployment readiness.

## 📊 Results and Visualizations

### Data Analysis Result

#### Call Failures
- Significant call failures in the range of 1-15 numbers.

#### Complaints Distribution
- Disproportionate distribution of complaints vs. non-complaints, requiring pre-processing before model training.

#### Subscription Commitment
- 75% of customers have subscriptions lasting 25-40 months, indicating moderate commitment.
- 16% are long-term subscribers.

#### Charge Amount
- 94% of customers fall into the low charge bracket (0 to 3 units).

#### Customer Activity
- 75% of customers are active; 25% are non-active.
- Non-active customers churn at a much higher rate (47%).

#### Tariff Plans
- 92% of customers are on Tariff Plan 1; only 8% prefer Tariff Plan 2.
- Tariff Plan 1 users show higher churn rates (20%) compared to Tariff Plan 2 users (1%).

#### Customer Value
- 67% of customers have a value below 500 units.
- Non-churned users have 4x the average customer value compared to churned users.

#### Churn Rate
- 15% of customers churned, with the imbalance needing adjustment during training.
- Age Group 4 has the highest churn rate at 20%.

#### Complaints and Churn
- Customers with complaints have an 83% churn rate, compared to 10% for those without complaints.

#### Age Group Analysis
- Age Group 3 has the highest number of complaints.
- Age Groups 2 and 3 have the highest customer value (~550 units).
- Age Groups 2, 3, and 4 have the highest churn rates (17%, 16%, and 20%).

#### Tariff Plans and Complaints
- Tariff Plan 1 and Tariff Plan 2 have similar complaint rates (7.5% vs 7.7%).
- Tariff Plan 1 users have longer subscription lengths but a higher churn rate.

### Data Analysis Conclusion
Efforts should focus on:
1. Addressing complaints and call failures to reduce churn.
2. Balancing the churn and complaint rate disparity during model training.
3. Targeting retention efforts towards non-active users and high-churn age groups.


### SHAP Summary Plot
This plot shows the impact of each feature on the model’s output.

![SHAP Summary Plot](https://github.com/VishShaji/Telecom-Churn-Prediction/blob/main/assets/shap.png)


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

The final chosen model is saved and the saved `.joblib` model file is loaded to verify correct working. It's performance is evaluated on 100 data rows.

- **Accuracy**: How well the model predicts overall.
- **Precision & Recall**: Measuring the trade-off between false positives and false negatives.
- **F1-Score**: Balances precision and recall.
- **ROC-AUC**: Area under the curve, measuring model performance.

   - **Accuracy**: `0.98`
   - **Precision**: `0.875`
   - **Recall**: `1`
   - **F1-Score**: `0.933`
   - **ROC-AUC**: `0.9983`


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
