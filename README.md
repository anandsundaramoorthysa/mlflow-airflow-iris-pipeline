# Iris ML Pipeline with Apache Airflow and MLflow

A beginner-friendly, fully local MLOps example that runs an Iris
classification pipeline **orchestrated by Apache Airflow** and **tracked by
MLflow**.

> **TL;DR** — Airflow decides *when* and *in what order* things run; MLflow
> records *what happened* (parameters, metrics, models) every time the
> pipeline runs. Everything runs on your own machine — no cloud, no
> Kubernetes, no paid services.

---

## Overview

The project trains two simple scikit-learn classifiers (Logistic Regression and
Random Forest) on the classic Iris dataset and logs the full experiment
history to MLflow. Airflow runs the workflow on a daily schedule (and on
demand); each Airflow task maps to one logical step of the ML workflow.

Each time the pipeline runs, MLflow creates **two runs** (one per model) under
the `iris-classification` experiment, storing:

- model **parameters**
- evaluation **metrics** (accuracy, precision, recall, F1 — macro averaged)
- the trained **model artifact**
- a **classification report** and a **confusion matrix**

This project was built and validated with **Apache Airflow 3.3.2** and
**MLflow 3.16.1** (see "Verified on" below).

## Problem statement

This project demonstrates the *plumbing* of an ML project: keeping the
"what does my workflow look like" (Airflow) separate from the
"what did each experiment produce" (MLflow). Both are independent but
integrated — Airflow *triggers* the ML code, and the ML code *writes* its
results to MLflow. Every run — triggered manually or by the daily schedule —
produces a fresh pair of MLflow runs you can compare side by side.

## Architecture

```text
                         AIRFLOW
                            |
                            |
                       DAG Scheduler
                            |
                            v
                  +-------------------+
                  |  load_data        |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |  preprocess_data  |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |  train_models     |
                  |                   |
                  | LogisticRegression|
                  | RandomForest      |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |  evaluate_models  |
                  +---------+---------+
                            |
                            |
                            v
                         MLFLOW
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
         Parameters      Metrics       Artifacts
                                         |
                                         v
                                    Trained Models
```

Data moves between tasks as simple CSV/joblib files under `data/` and
`models/` (no database, no magic).

## Responsibilities

| Tool         | Responsibility                                        |
|--------------|-------------------------------------------------------|
| Airflow      | Orchestration (scheduling, ordering, retries, logs)   |
| MLflow       | Experiment tracking (params, metrics, models, files)  |
| Scikit-learn | The machine learning itself                          |
| Pandas/NumPy | Data handling                                        |

`src/` holds all reusable ML logic; the DAG file only *orchestrates* it.

## Technologies

- Python 3.12 (3.10–3.14 supported by Airflow 3.3)
- Apache Airflow 3.3.2
- MLflow 3.16.1
- Scikit-learn 1.9.1
- Pandas 3.0.5 / NumPy 2.5.3
- Pytest 9.1.1

## Installation

> **Windows note:** Apache Airflow supports POSIX systems only. On Windows the
> supported path is **WSL2 (Ubuntu)** — this project was built and run inside
> WSL2. (On macOS/Linux, simply run the same commands in a normal shell.)

```bash
# 1. open a WSL shell and enter the project directory
cd /path/to/mlflow-airflow-project

# 2. create a virtual environment
python3 -m venv .venv
# (on some Ubuntu WSL images you need: sudo apt install python3.12-venv)

# 3. install dependencies with Airflow's official constraints file.
#    Replace "3.12" with your own `python3 -V` minor version.
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt"

# 4. configure environment (optional; see .env.example)
cp .env.example .env
```

## Running MLflow

```bash
cd /path/to/mlflow-airflow-project      # inside WSL
export MLFLOW_TRACKING_URI=http://localhost:5000

# start the tracking server + UI (SQLite backend, local artifact store)
.venv/bin/mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --default-artifact-root "$PWD/mlruns"
```

MLflow UI → <http://localhost:5000>

## Running Airflow

```bash
cd /path/to/mlflow-airflow-project      # inside WSL

export AIRFLOW_HOME=$PWD/.airflow
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL=5   # fast DAG discovery
export MLFLOW_TRACKING_URI=http://localhost:5000

# one process = db init + admin user + scheduler + UI
.venv/bin/airflow standalone
```

`airflow standalone` initializes the SQLite database, creates an **admin**
user with a randomly generated password, and starts the scheduler and the
web UI. The password is written to
`.airflow/simple_auth_manager_passwords.json.generated` (Airflow 3 uses the
SimpleAuthManager by default).

Airflow UI → <http://localhost:8080> (log in as `admin` with that password)

## Accessing the UIs

| Service  | URL                   | Credentials                                              |
|----------|-----------------------|----------------------------------------------------------|
| Airflow  | http://localhost:8080 | `admin` / password in `.airflow/simple_auth_manager_passwords.json.generated` |
| MLflow   | http://localhost:5000 | none                                                    |

## Running the pipeline

1. Start MLflow (above) and Airflow (above).
2. Open the Airflow UI → `iris_ml_pipeline` → **Trigger DAG** (▶ icon),
   or run it from the CLI:

   ```bash
   .venv/bin/airflow dags trigger iris_ml_pipeline
   ```

3. Watch the four tasks light up and turn green.

The DAG also runs automatically on a `@daily` schedule; because `catchup` is
`False` and the start date is in the past, the first scheduled run fires
shortly after you unpause the DAG.

To vary the experiments, tweak `MODEL_PARAMS` in `src/train.py`
(e.g. `C` for Logistic Regression, `n_estimators`/`max_depth` for Random
Forest) and trigger the DAG again — MLflow will show the new parameters in
fresh runs.

## MLflow experiment

Everything is logged under the **`iris-classification`** experiment. Open it
at <http://localhost:5000> and click any run to inspect:

- **Parameters** — `model_type`, `C`, `max_iter`, `random_state`
  (Logistic Regression); `model_type`, `n_estimators`, `max_depth`,
  `random_state` (Random Forest)
- **Metrics** — `accuracy`, `precision`, `recall`, `f1_score`
- **Artifacts** — the fitted `model` (MLflow format, loadable with
  `mlflow.pyfunc.load_model`), `classification_report.txt`,
  `confusion_matrix.json`

Each pipeline run produces **two runs** in MLflow, so the UI's experiment
page shows an ever-growing comparison table of Logistic Regression vs
Random Forest across runs.

## Airflow DAG

`iris_ml_pipeline` — 4 tasks, strictly sequential:

| Task              | What it does                                             |
|-------------------|----------------------------------------------------------|
| `load_data`       | Loads the Iris dataset, saves `data/iris.csv`            |
| `preprocess_data` | Stratified 80/20 split, saves `data/train.csv` + `data/test.csv` |
| `train_models`    | Fits both classifiers, saves them under `models/`        |
| `evaluate_models` | Scores both models, logs full MLflow runs (params, metrics, model, artifacts) |

Task failures raise; the DAG retries once and then marks the run `failed`
(any partially logged MLflow run is automatically marked `FAILED`).

## Results

- **Airflow** — task statuses and full logs in the Airflow UI (click a task →
  *Log*), also stored under `.airflow/logs/`.
- **MLflow** — every run's parameters, metrics, model and artifacts under
  *Experiments → iris-classification*.
- **Files** — `data/` (CSVs) and `models/` (joblib + params JSON) are
  overwritten on every run.

## Troubleshooting

- **`airflow` not found** — you are not inside the venv; use
  `.venv/bin/...` or `source .venv/bin/activate`.
- **`python3 -m venv` fails on WSL (ensurepip not available)** — install the
  venv module: `sudo apt-get install python3.12-venv`.
- **"Untrusted types found" when logging an `sklearn` model** — MLflow 3 uses
  the `skops` serializer which whitelists types. This repo already passes
  `skops_trusted_types=["sklearn.tree._tree.Tree"]` in `src/evaluate.py`.
- **DAG does not appear in the UI** — DAG discovery polls the folder every
  `AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL` seconds; wait a few seconds or
  check `airflow dags list-import-errors` for a parse error.
- **Airflow runs but MLflow has no runs** — the Airflow process must see
  `MLFLOW_TRACKING_URI=http://localhost:5000`; re-export it **before** starting
  `airflow standalone`.
- **Ports taken** — change MLflow's `--port` and Airflow's port (default
  8080). `ss -tln` in WSL shows listeners.
- **Running on native Windows PowerShell fails** — Airflow requires a POSIX
  OS; use WSL2 (this project is validated there).
- **Tests are slow on Windows-mounted drives** — WSL reads through `/mnt/e/...`
  are slower than the native ext4 filesystem; this is expected.

## Verified on

- Windows 11 + **WSL2 Ubuntu 24.04**, Python 3.12.3
- `apache-airflow==3.3.2`, `mlflow==3.16.1`, `scikit-learn==1.9.1`,
  `pandas==3.0.5`, `numpy==2.5.3`, `pytest==9.1.1`
- Airflow: SQLite metadata DB, `LocalExecutor`, `airflow standalone`
- MLflow: `sqlite:///mlflow.db` tracking store + `./mlruns` artifact store
- 3 full pipeline runs → 6 MLflow runs, all `FINISHED`; `pytest` 11/11 green

## Tests

```bash
.venv/bin/python -m pytest          # from the project root (inside WSL)
```

The suite covers dataset loading/shape, reproducible stratified splitting,
training + prediction, metric bounds, artifact content, and DAG structure
(import + task dependencies).

## Future improvements (out of scope here)

- MLflow **Model Registry** and automated model promotion
- Cloud / Docker / Kubernetes deployment
- CI/CD for training and deployment
- A production-grade Airflow database (PostgreSQL/MySQL)
- Monitoring, alerting and drift detection

Intentionally left out to keep the project local and beginner-friendly.

## Project layout

```text
mlflow-airflow-project/
├── dags/
│   └── iris_ml_pipeline.py     # Airflow DAG (orchestration only)
├── src/
│   ├── __init__.py
│   ├── config.py               # paths + environment config (.env)
│   ├── data.py                 # loading / splitting
│   ├── train.py                # model registry + training
│   └── evaluate.py             # metrics + MLflow logging
├── tests/
│   └── test_pipeline.py
├── data/                       # generated CSVs (gitignored)
├── models/                     # generated joblib files (gitignored)
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```