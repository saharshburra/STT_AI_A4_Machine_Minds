**Huggingface Spaces Link :** https://huggingface.co/spaces/manoz-037/UrbanNest-Rent-Prediction

# Assignment 4: PropTech Startup Strategy - Rent Prediction Pipeline

**Course**: Software Tools and Techniques for AI  
**Total Marks**: 20

## Problem Statement: UrbanNest Analytics

Built a complete rent prediction pipeline for Mumbai, Pune, Delhi, and Hisar that includes model optimization, experiment tracking, Streamlit serving, Docker packaging, and Hugging Face deployment.

## Dataset

- Training set: `Dataset/train.csv` (11,128 rows)
- Test set: `Dataset/test.csv` (2,782 rows)
- Target: `price`
- Categorical features encoded for training and UI reuse: `location`, `city`, `Status`, `property_type`

---

## Tasks & Mark Distribution

### Task 1: Data Preprocessing, Optimization & Tracking (8 Marks)

**What we did**

- Used all required features and encoded categorical columns with label encoders.
- Removed duplicate rows in train/test and filtered train rows that overlapped with test feature vectors before tuning, so the reported MAE is stricter and less optimistic.
- Saved encoder artifacts and mappings for frontend reuse in `models/label_encoders.pkl` and `models/label_mappings.json`.
- Saved a data-quality audit in `models/data_quality_report.json` to document the cleaning step.
- Compared 3 tuning strategies on `RandomForestRegressor` with 5-fold CV:
  - Grid Search (`GridSearchCV`) with 60 exact combinations.
  - Random Search (`RandomizedSearchCV`) for 60 trials.
  - Bayesian Optimization (Optuna) for 60 trials.
- Logged experiment runs and comparisons using Trackio.
- Retrained best model on full train set and reported final MAE on untouched test set.

**Results**

| Method            | Best CV MAE | Time (sec) | Best Parameters                                       |
| ----------------- | ----------: | ---------: | ----------------------------------------------------- |
| Grid Search       |    15496.30 |     253.21 | `max_depth=30, min_samples_split=2, n_estimators=150` |
| Random Search     |    15474.49 |     230.41 | `max_depth=30, min_samples_split=3, n_estimators=183` |
| Bayesian (Optuna) |    15470.25 |     365.37 | `max_depth=28, min_samples_split=2, n_estimators=177` |

- Best accuracy method: Bayesian (Optuna)
- Best compute-efficiency method (time): Random Search
- Final test MAE: 15057.30

**How "best compute-efficiency" was decided**

- All three methods were run under the same fair budget: 5-fold CV and 60 evaluations/trials each.
- We define compute-efficiency as the method with the lowest wall-clock tuning time under this same budget.
- From this run: Random Search = 230.41s, Grid Search = 253.21s, Bayesian (Optuna) = 365.37s.
- Therefore, Random Search is the most compute-efficient for this experiment setup.
- Note: this is separate from best accuracy, where Bayesian achieved the lowest CV MAE.

**Why the MAE increased after cleaning**

- The cleaned run removes duplicate examples and train rows that exactly matched test feature vectors.
- That makes the evaluation more realistic, because the model can no longer benefit from memorizing repeated or near-identical rows.
- The higher MAE is therefore expected and preferable for viva defense.

**Generated outputs**

- Notebook: [train.ipynb](train.ipynb)
- Model and encoders: [models/best_rf_model.pkl](models/best_rf_model.pkl), [models/label_encoders.pkl](models/label_encoders.pkl), [models/training_metadata.json](models/training_metadata.json), [models/data_quality_report.json](models/data_quality_report.json)
- Plots: [plots/trials_vs_error.png](plots/trials_vs_error.png), [plots/optuna_hyperparameter_space.png](plots/optuna_hyperparameter_space.png)
- Optuna plot note: `optuna_hyperparameter_space.png` is generated using `optuna.visualization.plot_contour` (fallback: `plot_optimization_history` if contour export fails).
- Trackio evidence: [screenshots/trackio_dashboard.png](screenshots/trackio_dashboard.png), [screenshots/trackio_dashboard_media&tables.png](screenshots/trackio_dashboard_media&tables.png)

**Reproducibility commands (Task 1)**

```bash
# 1) Install Git LFS and clone
git lfs install
git clone <repo-url>
cd A4
git lfs pull

# 2) Setup environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Re-run notebook end-to-end
jupyter nbconvert --to notebook --execute train.ipynb --output train.executed.ipynb
```

---

### Task 2: Web Application Development (4 Marks)

**What we did**

- Built a standalone Streamlit UI in [app.py](app.py).
- Added inputs for all prediction features using `st.selectbox` and `st.number_input`.
- Loaded saved model and encoders with `pickle.load`.
- Encoded user inputs and predicted rent on `Predict Rent` button click.
- Displayed output in-app using `st.success`.
- Aligned the BHK input with the binary encoding used by the trained dataset and model.

**Generated outputs**

- App source: [app.py](app.py)
- Dependencies: [requirements.txt](requirements.txt)

**Reproducibility commands (Task 2)**

```bash
source .venv/bin/activate
streamlit run app.py
```

Open in browser: http://localhost:8501

---

### Task 3: Docker Containerization & Networking (4 Marks)

**What we did**

- Containerized the Streamlit app with a root-level [Dockerfile](Dockerfile).
- Used Python base image, installed dependencies, exposed port `8501`, and set Streamlit run command.
- Validated local port forwarding from host to container.

**Generated outputs**

- Docker config: [Dockerfile](Dockerfile)
- Build proof: [screenshots/docker_build.png](screenshots/docker_build.png)
- Running container proof: [screenshots/docker_ps.png](screenshots/docker_ps.png)
- Browser proof on localhost: [screenshots/streamlit_working.png](screenshots/streamlit_working.png)

**Reproducibility commands (Task 3)**

```bash
docker build -t urbannest-rent .
docker run --rm -p 8501:8501 urbannest-rent

# In another terminal (verification)
docker ps
```

Open in browser: http://localhost:8501

---

### Task 4: Cloud Deployment via Hugging Face Spaces (4 Marks)

**What we did**

- Deployed the Streamlit app on Hugging Face as a Docker Space using the repository Dockerfile.
- Verified public inference endpoint is accessible.

**Deployment URL**

- https://huggingface.co/spaces/manoz-037/UrbanNest-Rent-Prediction

## Limitations

- Unseen categorical values in notebook test preprocessing are mapped to `-1` (especially for unseen `location` values). This is a safe fallback, but performance on completely new localities may be lower than on known localities.
- The Streamlit UI reduces this issue by restricting user selections to known encoded classes and auto-linking city with location.
- A small number of coordinate anomalies may exist in the source dataset. City-aware latitude/longitude bounds are applied in the UI, but model quality still depends on input data quality.
- Compute-efficiency conclusions are based on this assignment's fixed budget setup (5-fold CV and 60 evaluations per method).

## Git LFS Notice

This repository uses Git LFS for model binaries (`.pkl`).

```bash
git lfs install
git lfs pull
```
