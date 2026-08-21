# Heart Disease Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue)
![MLflow](https://img.shields.io/badge/MLflow-experiment_tracking-blue)
![DVC](https://img.shields.io/badge/DVC-data_versioning-orange)
![Monitoring](https://img.shields.io/badge/Monitoring-Evidently-green)


## Overview

This project implements an end-to-end MLOps pipeline for predicting the presence of heart disease using clinical patient data. The goal is not only to train a machine learning model, but also to demonstrate production-ready machine learning practices including data versioning, experiment tracking, automated testing, continuous integration, and model monitoring.

The project uses a Heart Disease dataset containing demographic, clinical, and diagnostic measurements to classify whether a patient has heart disease. The pipeline is designed to be reproducible, maintainable, and extensible, following industry-standard MLOps workflows.

---

## Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early identification of individuals at risk can support timely intervention and improve health outcomes.

This project builds a classification model that predicts the presence of heart disease based on patient characteristics such as age, cholesterol levels, blood pressure, chest pain type, electrocardiogram results, and other clinical indicators.

---

## Dataset

### Source

Heart Disease Dataset (UCI Machine Learning Repository)

### Target Variable

**target**

* 0 = No heart disease
* 1 = Presence of heart disease

### Features

| Feature  | Description                                    |
| -------- | ---------------------------------------------- |
| age      | Age in years                                   |
| sex      | Biological sex                                 |
| cp       | Chest pain type                                |
| trestbps | Resting blood pressure                         |
| chol     | Serum cholesterol                              |
| fbs      | Fasting blood sugar                            |
| restecg  | Resting electrocardiographic results           |
| thalach  | Maximum heart rate achieved                    |
| exang    | Exercise-induced angina                        |
| oldpeak  | ST depression induced by exercise              |
| slope    | Slope of peak exercise ST segment              |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia category                           |

---

## Project Structure

```text
heart-disease-prediction/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── heart_disease.csv
│
│
├── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluation.py
│   ├── compare_experiments.py
│   ├── monitor_drift.py
│   ├── run_experiments.py          
│   ├── llm_interface.py            
│   └── app.py                      
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   ├── test_model_validation.py
│   └── test_interface.py           
│
├── knowledge_base/                 
│   ├── age.md
│   ├── sex.md
│   ├── cp.md
│   ├── trestbps.md
│   ├── chol.md
│   ├── fbs.md
│   ├── restecg.md
│   ├── thalach.md
│   ├── exang.md
│   ├── oldpeak.md
│   ├── slope.md
│   ├── ca.md
│   ├── thal.md
│   └── heart_disease_overview.md
│
├── models/                          
│   ├── best_model.joblib
│   ├── feature_columns.json
│   ├── feature_defaults.json
│   └── best_run_info.json
│
├── requirements.txt
├── README.md
├── .env                     
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* MLflow
* DVC
* Pytest
* GitHub Actions
* Evidently
* YAML
* Streamlit 
* OpenAI-compatible SDK
* python-dotenv 

---

## MLOps Components

### Data Versioning (DVC)

The dataset is tracked using DVC rather than Git to enable reproducible experimentation and efficient version control of large files.

### Experiment Tracking (MLflow)

Each training run logs:

* Model hyperparameters
* Dataset version
* Evaluation metrics
* Trained model artifacts

Experiments can be compared programmatically using `compare_experiments.py`.

### Automated Testing (Pytest)

The project includes:

#### Preprocessing Tests

* Missing value handling
* Categorical encoding validation
* Data integrity checks
* Error handling validation

#### Data Validation Tests

* Required columns exist
* Target values are valid
* Numeric feature ranges are reasonable

#### Model Validation Tests

* Prediction shape verification
* Minimum performance threshold validation

#### Interface Tests

* LLM response parsing (well-formed JSON, markdown-fenced JSON, malformed input)
* Missing critical vs. auxiliary feature detection
* Clarifying-question and out-of-scope message generation
* Full pipeline behavior with a mocked LLM client (no live API needed to test)

### Continuous Integration (GitHub Actions)

The CI/CD pipeline automatically:

1. Installs dependencies
2. Executes the test suite
3. Trains the model
4. Verifies model performance requirements

### Drift Monitoring (Evidently)

The monitoring pipeline compares reference training data against simulated production data to detect feature drift and identify when retraining may be necessary.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd heart-disease-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Training

Train the model:

```bash
python src/train.py
```

The training script:

* Loads configuration values from `config.yaml`
* Preprocesses the data
* Trains the model
* Evaluates performance
* Logs metrics and artifacts to MLflow

---

### Running Multiple Experiments 

`src/train.py` above logs a single run from `config.yaml`. To generate the
5+ meaningfully different runs used for model selection, run:

```bash
python src/run_experiments.py
```

This logs 7 configurations spanning five algorithm families — three
RandomForest variants, GradientBoosting, LogisticRegression, SVC, and KNN —
each as its own MLflow run with full hyperparameters, metrics, and the
model artifact attached.

---

## Running Tests

Run all tests:

```bash
pytest tests/ -v
```

---

## MLflow Experiment Tracking

Launch the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then navigate to:

```text
http://localhost:5000
```

View:

* Experiment runs
* Parameters
* Metrics
* Model artifacts

---

## Comparing Experiments

Identify the best-performing experiment:

```bash
python src/compare_experiments.py
```

The script queries MLflow runs and returns the highest-performing model based on the primary evaluation metric.

*(Update: this script now also exports the winning model, its feature
column order, and population-median feature defaults to `models/`, so the
LLM interface below can load it directly without an MLflow dependency at
inference time.)*

### Results from the 7-run comparison

The best-performing configuration was a RandomForest with 300 trees and max
depth 12 (`rf_deep_wide`):

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 0.918  |
| F1       | 0.915  |
| ROC AUC  | 0.958  |

---

## Drift Monitoring

Run drift detection:

```bash
python src/monitor_drift.py
```

Outputs:

* Drift summary in terminal
* HTML drift report saved to `reports/`

---

## LLM-Powered Interface

On top of the trained model, the app now includes a natural-language
interface (`src/llm_interface.py` + `src/app.py`) so a user can describe a
patient in plain English instead of filling out a form:

> *"I'm a 55-year-old male with atypical angina chest pain, cholesterol of 240, > resting blood pressure 150, max heart rate 130, exercise-induced angina, and
> an ST depression of 2.3"*

The pipeline:

1. **Parse** — an LLM call extracts the 13 clinical features from the
   message into structured JSON.
2. **Validate** — features are split into **critical** (age, sex, cp,
   trestbps, chol, thalach, exang, oldpeak — the app asks a clarifying
   question if any are missing rather than guessing) and **auxiliary**
   (fbs, restecg, slope, ca, thal — fields a user may not know, which fall
   back to population-median defaults, always disclosed as an assumption).
3. **Scope check** — out-of-scope requests (e.g. asking for medication
   advice) get an explanation of what the tool can/can't do instead of a
   garbage prediction.
4. **Predict** — the real trained model (loaded from `models/best_model.joblib`)
   runs inference.
5. **Explain** — top contributing factors are ranked by
   `feature importance × deviation from population median`, optionally
   grounded with `knowledge_base/*.md` content, and a second LLM call turns
   this into a clear, caveated explanation.

Two different models are used for steps 1 and 5 — a cheap model for
mechanical JSON extraction, and a stronger model for the explanation the
user actually reads.

### LLM Provider Setup

Copy `.env.example` to `.env` and fill in your key. Default provider is
**Nebius Token Factory**:

```
LLM_PROVIDER=nebius
NEBIUS_API_KEY=your_key_here
NEBIUS_PARSE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
NEBIUS_EXPLAIN_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

To use OpenAI instead:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_PARSE_MODEL=gpt-4o-mini
OPENAI_EXPLAIN_MODEL=gpt-4o-mini
```

**Never commit `.env`** — it's already excluded via `.gitignore`.

### Running the LLM Interface

```bash
streamlit run src/app.py
```

Opens a chat UI in the browser. Suggested test queries for the demo:

* **Complete query:** *"I'm a 55-year-old male with atypical angina chest pain, cholesterol of 240, > resting blood pressure 150, max heart rate 130, exercise-induced angina, and an ST depression of 2.3"*
* **Incomplete query** (tests clarifying-question handling): *"I'm 55 and
  male"*
* **Out-of-scope query** (tests scope handling): *"What medication should I
  take for my heart?"*

---

## Future Improvements

* Hyperparameter optimization using Optuna
* Model deployment with FastAPI
* Containerization with Docker
* Cloud deployment using AWS
* Automated retraining workflows
* Real-time monitoring dashboards
* SHAP-based explainability instead of the current importance-based heuristic

---

# Data Drift Analysis

## Summary

A drift analysis was performed using Evidently to compare the reference dataset against a current dataset. Several features exhibited measurable drift.

## Drifted Features

* age
* cholesterol (chol)
* resting blood pressure (trestbps)

## Potential Impact

Changes in the distribution of these features may reduce model performance because the model was trained on historical patterns that may no longer represent the current patient population.

Potential effects include:

* Reduced prediction accuracy
* Lower F1 score
* Increased false positive and false negative rates

## Recommended Action

The model should continue to be monitored regularly. If drift persists or model performance declines below acceptable thresholds, retraining using more recent data is recommended.

## Conclusion

Although the model remains operational, continued monitoring and periodic retraining are advised to ensure reliable heart disease predictions.


## Author

**Chinweoke Stephanie Okonkwo**

Data Science | Machine Learning | MLOps | Healthcare Analytics
