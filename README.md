### Student Score Prediction Web App (Flask + Machine Learning)

A resume-ready end-to-end Machine Learning project that predicts a student’s score using demographic + academic inputs, served through a Flask web application using a trained model and preprocessing pipeline.

---

## Overview

This project predicts a student’s score based on inputs such as gender, race/ethnicity, parental education level, lunch type, test preparation status, reading score, and writing score.  
It uses a saved **preprocessor** and **trained model** for inference and displays the predicted score on the web UI.

---

## Features

- Flask web app with clean UI (HTML templates)
- ML prediction using saved artifacts:
  - `model.pkl`
  - `preprocessor.pkl`
- Input validation through HTML constraints
- DataFrame-based inference pipeline
- Result displayed on the same page after prediction

---

## Tech Stack

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- Dill

---

## Project Structure

```text
ML-Project/
│
├── app.py
├── requirements.txt
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── artifact/
│   ├── model.pkl
│   └── preprocessor.pkl
│
└── src/
    ├── pipelines/
    │   └── Predict_Pipeline.py
    ├── Utils.py
    ├── Exception.py
    └── (other components...)
```
---

## Routes

| Route | Method | Purpose |
|------|--------|---------|
| `/` | GET | Landing page (`index.html`) |
| `/predictdata` | GET | Prediction form page (`home.html`) |
| `/predictdata` | POST | Predict score and render results on `home.html` |

---

## Inputs

The UI collects these fields:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

---

## Setup

### 1) Clone the repository
```bash
git clone <your-repo-url>
cd ML-Project
```

## 2) Create and activate a virtual environment (Windows)
```bash
    python -m venv venv
    venv\Scripts\activate
```

## 3) Install dependencies
```bash
pip install -r requirements.txt

Run the App
python app.py
```
Open:
http://127.0.0.1:5000/

---

## Artifacts

This app expects these files:

artifact/model.pkl
artifact/preprocessor.pkl

They are loaded using src/Utils.py -> load_object().

---

## How It Works (Prediction Flow)

1. User opens `/predictdata`
2. User fills the form and submits (POST)
3. Flask collects values via `request.form.get(...)`
4. `CustomData` converts inputs into a Pandas DataFrame
5. `preprocessor.pkl` transforms the DataFrame
6. `model.pkl` predicts the score
7. Result is displayed in `home.html`

---

## Notes for Deployment

- Ensure `artifact/` contains both `model.pkl` and `preprocessor.pkl`
- Pin package versions in `requirements.txt` to avoid `.pkl` incompatibilities
- Recommended: retrain + resave artifacts if Python/scikit-learn versions change

---

## Recommended `requirements.txt` (example)

```txt
flask
numpy
pandas
scikit-learn==1.3.2
dill
```

---

## Common Issues & Fixes

# 1) Form key mismatch / CustomData argument error

Example:
TypeError: __init__() got an unexpected keyword argument '...'

Fix:
Ensure keys match across home.html, app.py, CustomData, and DataFrame column names.

# 2) Scikit-learn version mismatch while loading .pkl

Example:
Trying to unpickle estimator ... from version X when using version Y

Fix options:
Use the same scikit-learn version used during training, OR
Re-train and re-save model.pkl and preprocessor.pkl using your current environment.

---

## Resume Bullet

Built an end-to-end Student Score Prediction web app using Flask + Scikit-learn, integrating preprocessing pipelines, artifact-based model inference, and a clean HTML UI for real-time predictions.

---

## License

Educational/portfolio use. Add a LICENSE file if open-sourcing.

---

## Author

Karamjodh Singh
GitHub: <https://github.com/Karamjodh>
LinkedIn: <https://www.linkedin.com/in/karamjodh-singh/>

---
