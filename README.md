# 🏠 Housing Price Data Mining

A complete data mining and machine learning project for predicting median housing prices through exploratory data analysis, preprocessing, feature engineering, regression modeling, and hyperparameter optimization.

## 📌 Overview

This project implements an end-to-end data mining workflow for housing price prediction—a classic regression problem. It progresses from raw data exploration through preprocessing and model training to final evaluation on unseen test data, emphasizing both predictive accuracy and understanding the underlying data.

### Complete Workflow

- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Feature engineering and scaling
- Categorical encoding
- Reusable preprocessing pipelines
- Multiple regression model comparison
- Cross-validation
- Hyperparameter optimization with Grid Search
- Final evaluation on held-out test data

## 🎯 Objectives

- Explore and understand housing dataset characteristics
- Identify relationships between attributes and house prices
- Handle missing and categorical data appropriately
- Engineer meaningful features from existing attributes
- Build and compare multiple regression models
- Evaluate performance using cross-validation
- Optimize the best model with Grid Search
- Assess final model on unseen test data

## 📊 Dataset

The target variable is **`median_house_value`** — the median house value for a given district.

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude of the district |
| `latitude` | Geographic latitude of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Total population in the district |
| `households` | Number of households |
| `median_income` | Median income of households |
| `ocean_proximity` | Categorical feature: proximity to ocean |
| `median_house_value` | **Target variable** |

## 🔍 Exploratory Data Analysis

- **Data Inspection**: Structure, data types, and non-null counts via `housing.info()` and `housing.shape`
- **Distribution Analysis**: Histograms to identify skewed distributions, feature scales, and outliers
- **Correlation Analysis**: Correlation matrix to reveal linear relationships with the target
- **Scatter Plots**: Visualizing `median_income` vs. `median_house_value` patterns

## 🧹 Data Preprocessing

- **Train/Test Split**: 80/20 split with `random_state=42`
- **Missing Value Handling**: Median imputation using `SimpleImputer` (robust to outliers)
- **Feature Scaling**: Standardization with `StandardScaler` (mean=0, std=1)
- **Categorical Encoding**: One-Hot Encoding for `ocean_proximity` using `OneHotEncoder`

## ⚙️ Feature Engineering

Three engineered features are created from existing attributes:

| Feature | Formula | Description |
|---|---|---|
| `rooms_per_household` | `total_rooms / households` | Average rooms per household |
| `population_per_household` | `population / households` | Average people per household |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Bedroom-to-room ratio |

## 🔄 Preprocessing Pipeline

A reusable Scikit-learn pipeline combining:

- **Numerical Pipeline**: Feature selection → Imputation → Feature engineering → Standardization
- **Categorical Pipeline**: Feature selection → One-Hot Encoding
- **Combined**: Both pipelines merged using `FeatureUnion`

## 🤖 Regression Models

| Model | Description |
|---|---|
| **Linear Regression** | Baseline model for comparison |
| **Decision Tree Regressor** | Captures non-linear relationships |
| **Random Forest Regressor** | Ensemble method for robust predictions |

## 🔁 Cross-Validation

- **Method**: 10-fold cross-validation
- **Metric**: Negative Mean Squared Error → Root Mean Squared Error (RMSE)
- **Analysis**: Per-fold RMSE, mean RMSE, and standard deviation

## 🔧 Hyperparameter Optimization

Grid Search with 5-fold cross-validation on Random Forest:

```python
param_grid = [
    {
        "n_estimators": [3, 4, 6, 10, 30],
        "max_features": [2, 6, 8, 15]
    }
]
```

The best configuration is selected as the final model.

🧪 Final Model Evaluation

1. Extract best estimator: final_model = grid_search.best_estimator_
2. Transform test set using the preprocessing pipeline
3. Generate predictions on unseen data
4. Compute final RMSE: final_rmse = np.sqrt(mean_squared_error(y, final_predictions))

📈 Evaluation Strategy

```
Raw Dataset
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Train/Test Split
    │
    ▼
Data Preprocessing
    ├── Missing Value Imputation
    ├── Feature Engineering
    ├── Feature Scaling
    └── One-Hot Encoding
    │
    ▼
Regression Models
    ├── Linear Regression
    ├── Decision Tree Regression
    └── Random Forest Regression
    │
    ▼
Cross-Validation
    │
    ▼
Model Comparison
    │
    ▼
Grid Search
    │
    ▼
Best Random Forest Model
    │
    ▼
Final Evaluation on Test Set
```

📁 Project Structure

```
housing-price-data-mining/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── housing_data_mining.ipynb
│
├── docs/
│   └── ...
│
├── README.md
├── requirements.txt
└── .gitignore
```

🛠️ Technologies & Libraries

Core: Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook

Key Scikit-learn Components:
train_test_split, SimpleImputer, StandardScaler, OneHotEncoder, Pipeline, FeatureUnion, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, cross_val_score, GridSearchCV, mean_squared_error



📦 Requirements

```
pandas
numpy
matplotlib
scikit-learn
jupyter
```

📊 Results

Model Evaluation Method
Linear Regression 10-Fold Cross-Validation
Decision Tree Regression 10-Fold Cross-Validation
Random Forest Regression 10-Fold Cross-Validation
Tuned Random Forest Grid Search + 5-Fold CV
Final Model Unseen Test Set

Note: Exact numerical results should be added after final model evaluation is completed. Performance is reported using RMSE—lower values indicate better predictive accuracy.

🎓 Key Learning Outcomes

· Understanding and exploring real-world tabular data
· Identifying relationships between variables
· Handling missing values and categorical features
· Performing feature engineering
· Scaling numerical features
· Designing reusable preprocessing pipelines
· Comparing different regression algorithms
· Using cross-validation for reliable evaluation
· Performing hyperparameter optimization
· Evaluating final models on unseen data

🔮 Future Improvements

· Testing additional regression algorithms
· More extensive hyperparameter optimization
· Feature selection techniques
· Comparing different imputation strategies
· Additional engineered features
· Residual and error analysis
· Visualizing predicted vs. actual values
· Additional metrics: MAE, $R^2$
· Experiment tracking and automated model comparison


Computer Engineering | Data Mining | Machine Learning | Deep Learning

📄 License

This project is intended for educational and research purposes.

```
