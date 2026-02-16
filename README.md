# Telecom Customer Churn Prediction with AutoML

This project uses AutoML tools (PyCaret and TPOT) to build a churn prediction model for a telecom company. Given customer data like tenure, contract type, payment method, and charges, the model predicts whether a customer is likely to churn.

## Dataset

The dataset contains 7,043 telecom customers with 8 features:
- **tenure**: months with the company
- **PhoneService**: yes/no
- **Contract**: month-to-month, one year, or two year
- **PaymentMethod**: electronic check, mailed check, bank transfer, or credit card
- **MonthlyCharges** and **TotalCharges**: billing amounts
- **Churn**: the target variable (yes/no)

About 26.5% of customers churned, making this a moderately imbalanced classification problem.

## Approach

### Metric Choice: AUC

We used **AUC (Area Under the ROC Curve)** instead of accuracy because:
- A model that always predicts "no churn" would get ~73.5% accuracy — not useful
- AUC measures how well the model separates churners from non-churners across all possible thresholds
- It's the right metric when you don't know the exact business cost of false positives vs. false negatives

### PyCaret

PyCaret compared many classifiers automatically and selected the best one based on AUC with 10-fold cross-validation. The model was saved as a pickle file that includes the full preprocessing pipeline.

### TPOT Comparison

TPOT uses genetic programming to evolve machine learning pipelines. It took a few minutes to run and explored creative pipeline combinations. Both tools produced strong models with comparable AUC scores.

## Results

Predictions on 5 new customers (true values: `[1, 0, 0, 1, 0]`):
- The model produces churn probabilities for each customer
- Each prediction includes a percentile ranking showing where it falls relative to the training data distribution

## How to Run

### Setup

This project requires **Python 3.10** (PyCaret compatibility). Create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Make Predictions

```bash
# With a specific file
python inference_churn.py new_churn_data.csv

# Or let it prompt you for a file path
python inference_churn.py
```

The script accepts both label-encoded data (like `new_churn_data.csv`) and raw unmodified data (with string categories like the original dataset). It auto-detects the format.

## Files

| File | Description |
|------|-------------|
| `Week_5_assignment_Santiago.ipynb` | Main notebook with exploration, model training, and evaluation |
| `inference_churn.py` | Python module with `ChurnPredictor` class for making predictions |
| `churn_pycaret_model.pkl` | Saved PyCaret model (preprocessing pipeline + classifier) |
| `train_prob_distribution.pkl` | Training prediction distribution for percentile calculations |
| `churn_tpot_pipeline.pkl` | Saved TPOT pipeline for comparison |
| `churn_data_week1.csv` | Original training data |
| `new_churn_data.csv` | Test data for predictions |
| `requirements.txt` | Python dependencies |

## Optional Features Implemented

- **Percentile rankings**: each prediction shows where it falls in the training distribution (e.g. 90th percentile = higher churn risk than 90% of training data)
- **TPOT comparison**: side-by-side evaluation of PyCaret vs TPOT
- **Class-based module**: `ChurnPredictor` class in `inference_churn.py`
- **User input**: accepts file path from command line or interactive prompt
- **Raw data handling**: `preprocess_raw_data()` method handles unmodified churn data with string categories and missing values
