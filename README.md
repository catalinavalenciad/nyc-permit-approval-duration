# NYC Construction Permit Approval Duration

## Overview
This project builds a regression modeling pipeline to predict how long it takes for a construction permit to be approved by the NYC Department of Buildings. Using 100,000 records pulled live from the NYC Open Data API, the analysis covers the full modeling workflow from data acquisition and feature engineering through baseline regression, regularization, and tree-based methods.

The project is structured as an end-to-end **regression pipeline** applied to a real-world AEC (Architecture, Engineering, and Construction) dataset, with an emphasis on transparent model comparison and honest evaluation of predictive limitations.

A central finding of this work is that **available categorical features are insufficient** to reliably predict permit approval duration, and the analysis surfaces what additional data would be needed to improve performance.

## Business Problem
Permit approval timelines are a core source of uncertainty in construction project planning:

- Delays in permit issuance directly affect project schedules and costs
- Approval duration varies widely across job types, boroughs, and building classifications

While this project does not encode explicit cost asymmetry, it provides a structured, data-driven baseline for understanding what drives approval time and where predictive modeling falls short with currently available features.

The goal is to identify whether permit metadata alone can predict approval duration, and to quantify the gap between model performance and practical usefulness.

## Data Source
- **NYC DOB Permit Issuance Dataset**
  Publicly available through **NYC Open Data**
  https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a

**Dataset characteristics:**
- 100,000 records pulled via the Socrata API (Dataset ID: `ipu4-2q9a`)
- Response variable: `approval_duration` (days from filing to issuance)
- 30 encoded predictors after feature engineering
- Predictors include: job type, work type, borough, building type, permit type, and residential classification

## Tech Stack
- **Language:** Python
- **Environment:** Jupyter Notebook
- **Key Libraries:**
  - `sodapy` (NYC Open Data Socrata API client)
  - `pandas` (data manipulation)
  - `scikit-learn` (regression models, cross-validation, metrics)
  - `matplotlib` (visualization)

## Methodology

### 1. Data Acquisition and Overview
- Connected to the NYC Open Data API via `sodapy`
- Pulled 100,000 permit records without manual download
- Inspected dataset shape, column types, and missingness patterns

### 2. Feature Engineering and Cleaning
- Parsed `filing_date` and `issuance_date` as datetime objects
- Computed `approval_duration` as the difference in days
- Filtered out same-day approvals, negative durations, and records missing key fields
- Applied one-hot encoding to categorical predictors

### 3. Baseline Regression
- Fit an OLS linear regression model as a performance benchmark
- Evaluated on a held-out test set using RMSE and R²
- Interpreted coefficients to identify the most influential predictors

### 4. Regularization Models
- Applied **Ridge**, **Lasso**, and **Elastic Net** regression with cross-validated alpha selection
- Used Lasso's sparsity property for implicit feature selection
- Compared performance against the OLS baseline

### 5. Tree-Based Models
- Fit a **Random Forest** regressor to capture potential non-linear relationships
- Examined feature importance scores across all 30 predictors
- Evaluated generalization performance on the same held-out test set

## Results & Key Insights

| Model | RMSE | R² |
|---|---|---|
| Baseline OLS | 51.95 | 0.0054 |
| Ridge | 51.94 | 0.0060 |
| Lasso | 51.90 | 0.0075 |
| Elastic Net | 51.90 | 0.0073 |
| Random Forest | 52.70 | −0.0234 |

**Selected model: Lasso** — best R², with only 12 of 30 predictors retained.

- No model explains more than 0.75% of the variance in approval duration.
- Permit approval time is not reliably determined by job type, borough, or permit type alone.
- Lasso identifies work type and permit type as the most informative features.
- Random Forest fails to generalize despite capturing non-linear relationships.
- Predictive performance is fundamentally limited by feature availability, not modeling choices.

## Repository Structure
```
nyc-permit-approval-duration/
├── data/
│   └── permit_model_data.csv

├── notebooks/
│   ├── 01_data_acquisition_and_overview.ipynb
│   ├── 02_feature_engineering_and_cleaning.ipynb
│   ├── 03_baseline_regression.ipynb
│   ├── 04_regularization_models.ipynb
│   ├── 05_tree_based_models.ipynb
│   └── 06_model_synthesis_and_conclusions.ipynb

├── docs/
│   ├── index.html
│   ├── 01_data_acquisition_and_overview.html
│   ├── 02_feature_engineering_and_cleaning.html
│   ├── 03_baseline_regression.html
│   ├── 04_regularization_models.html
│   ├── 05_tree_based_models.html
│   └── 06_model_synthesis_and_conclusions.html

├── README.md
└── .gitignore
```

## Notes
This project prioritizes methodological transparency and honest reporting over performance optimization.
The low R² values are a finding, not a flaw, they reflect a real limitation of the available data for this prediction task.

---

## Contact
- **Name:** Catalina Valencia
- **Email:** catalinavalenciad@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/catalina-valencia/
