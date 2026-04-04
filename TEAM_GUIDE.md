# Team Guide: How to Complete Your Task Chunk

This guide is for team members who want to run the project locally and complete their assigned part without breaking required submission format.

## 1. Project Structure You Must Preserve

Keep this structure intact:

- README.md
- requirements.txt
- train.ipynb
- app.py
- Dockerfile
- models/
- plots/
- Dataset/
- screenshots/

Do not rename required files or screenshot names.

## 2. Environment Setup (Windows-safe)

### First-time setup (everyone should do this)

Run from project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python` is not found, use the Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Task Ownership Guidance

Split work cleanly so each member owns one area.

- Member A: Task 1 (training notebook, tracking, artifacts, plots)
- Member B: Task 2 (Streamlit app inputs, model loading, prediction UI)
- Member C: Task 3 (Docker build/run and screenshots)
- Member D: Task 4 (Hugging Face deployment + README top URL)

If fewer members, combine adjacent tasks.

## 4. What Each Task Must Produce

## Task 1 (8 marks)

Required outputs:

- train.ipynb with preprocessing + Grid/Random/Bayesian tuning
- models/best_rf_model.pkl
- models/label_encoders.pkl
- models/label_mappings.json
- models/training_metadata.json
- plots/trials_vs_error.png
- plots/optuna_hyperparameter_space.png
- screenshots/trackio_dashboard.png

Notes:

- Use 5-fold CV.
- Use required search ranges:
  - n_estimators: 50 to 200
  - max_depth: 10 to 30
  - min_samples_split: 2 to 10
- Trackio screenshot should visibly show compared methods, params, and scores.

## Task 2 (4 marks)

Required outputs:

- app.py
- requirements.txt includes streamlit, scikit-learn, optuna, trackio (and any extra used package)

App behavior:

- Uses all required input features from dataset
- Loads model and encoders with pickle
- Applies preprocessing
- Predicts on button click
- Shows prediction result clearly

## Task 3 (4 marks)

Required outputs:

- Dockerfile
- screenshots/docker_build.png
- screenshots/docker_ps.png
- screenshots/streamlit_working.png

Expected commands:

```powershell
docker build -t rent-predictor .
docker run -p 8501:8501 rent-predictor
docker ps
```

Browser URL for screenshot:

- http://localhost:8501

## Task 4 (4 marks)

Required outputs:

- Public Hugging Face Space URL
- URL placed at top of README.md

Deployment requirement:

- Use Docker Space template
- Space must run and return predictions

## 5. Screenshot Naming Rules (Strict)

Use exactly these names in screenshots/:

- trackio_dashboard.png
- docker_build.png
- docker_ps.png
- streamlit_working.png

Extra screenshots are allowed but do not replace required names.

## 6. Quick Verification Before Push

Run these checks from project root:

```powershell
Get-ChildItem .\models
Get-ChildItem .\plots
Get-ChildItem .\screenshots
```

Confirm required files are present and correctly named.

## 7. Common Issues and Fixes

PowerShell activation blocked:

- Use .venv Python directly (Option A above)

trackio command not found:

- Use:

```powershell
.\.venv\Scripts\python.exe -m trackio.cli show --project "STT-A4-Rent-Prediction"
```

Media & Tables empty in Trackio:

- You only logged scalar metrics. Log a Trackio table if needed for clearer screenshot evidence.
