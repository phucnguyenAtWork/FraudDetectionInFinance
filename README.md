# Fraud Detection in Finance

A machine learning project to detect fraudulent credit card transactions using a highly imbalanced real-world dataset (~0.57% fraud rate). Multiple models are compared and optimized for fraud recall.

## Project Structure
## Dataset link
https://www.kaggle.com/datasets/kartik2112/fraud-detection
```
BDA/
├── README.md
├── Fraud_Detection/
│   ├── Fraud_Detection_Clean.ipynb   # Main notebook (all code & analysis)
│   ├── spark_fraud_model/            # Saved Apache Spark pipeline model
│   ├── presentation.html             # Plotly interactive presentation
│   └── fig_*.png                     # Generated visualizations
└── Real_Data_Fraud(Dataset)/
    ├── fraudTrain/fraudTrain.csv     # Training set (~1.3M transactions)
    └── fraudTest/fraudTest.csv       # Test set (~556K transactions)
```

## Dataset

Simulated credit card transaction data containing legitimate and fraudulent transactions.

**Key columns:** `amt`, `merchant`, `category`, `job`, `city_pop`, `lat/long`, `merch_lat/merch_long`, `dob`, `trans_date_trans_time`, `is_fraud` (target)

## How to Run

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/phucnguyenAtWork/FraudDetectionInFinance.git
   cd FraudDetectionInFinance
   ```

2. Install dependencies (handled in the first notebook cell, or manually):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn plotly pyspark
   ```

3. Make sure the dataset is placed under `Real_Data_Fraud(Dataset)/`:
   ```
   Real_Data_Fraud(Dataset)/
   ├── fraudTrain/fraudTrain.csv
   └── fraudTest/fraudTest.csv
   ```

4. Open and run the notebook:
   ```bash
   jupyter notebook "Fraud_Detection /Fraud_Detection_Clean.ipynb"
   ```

5. **Run all cells sequentially** from top to bottom. The notebook is self-contained — each section depends on the cells above it.

## Notebook Sections

| Section | Description |
|---------|-------------|
| **1. Setup** | Install packages and import libraries |
| **2. Data Loading & Cleaning** | Load CSVs, convert datetimes, drop unnecessary columns |
| **3. Missing Values & Duplicates** | Check data quality |
| **4. Exploratory Data Analysis (EDA)** | Distribution plots, fraud timelines, correlation heatmap, outlier analysis, fairness check by gender |
| **5. Feature Engineering** | Create `customer_age`, `distance_km`, `hour_of_day`, `trans_count_24h`, `trans_amt_24h`; encode categoricals; scale features; apply SMOTE for class balancing |
| **6. Model Training** | Train Random Forest and XGBoost with hyperparameter tuning (RandomizedSearchCV) |
| **7. Model Comparison** | Compare Dummy, Logistic Regression, RF, XGBoost, and Stacking Ensemble across Precision, Recall, F1, and AUC-ROC; PR/ROC curves; threshold analysis; confusion matrices; business cost analysis |
| **8. Big Data Scalability** | Apache Spark pipeline demo for large-scale deployment |
| **9. Fairness Check** | Error rate comparison across gender groups |

## Models Compared

| Model | Purpose |
|-------|---------|
| **Dummy Classifier** | Baseline (always predicts majority class) |
| **Logistic Regression** | Simple linear baseline |
| **Random Forest** | Ensemble of decision trees (with hyperparameter tuning) |
| **XGBoost** | Gradient boosting (with hyperparameter tuning) |
| **Stacking Ensemble** | Combines RF + XGBoost with Logistic Regression meta-learner |

## Engineered Features

| Feature | Description |
|---------|-------------|
| `customer_age` | Age of cardholder at transaction time |
| `distance_km` | Haversine distance between customer and merchant |
| `hour_of_day` | Hour when transaction occurred |
| `trans_count_24h` | Number of transactions by same card in last 24 hours |
| `trans_amt_24h` | Total amount spent by same card in last 24 hours |

## Key Techniques

- **SMOTE** oversampling to handle class imbalance (0.57% fraud)
- **StandardScaler** for feature normalization
- **RandomizedSearchCV** for hyperparameter tuning
- **Precision-Recall curves** (more informative than ROC for imbalanced data)
- **Decision threshold tuning** to optimize fraud recall
- **Business cost analysis** translating model metrics into dollar impact
