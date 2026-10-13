# DecodeLabs Project 2 — Data Classification Using AI

A beginner-friendly supervised machine learning project based on the Project 2 brief. It loads a small labeled dataset, splits it into training/testing sets, trains a classifier, evaluates it, and predicts a class for new measurements.

## Features
- Built-in Iris dataset (150 records; no separate CSV download required)
- 80/20 train-test split with a fixed random seed
- StandardScaler + K-Nearest Neighbors (KNN, 5 neighbors)
- Accuracy, confusion matrix, and per-class precision/recall/F1
- Flask web interface to enter measurements and view a prediction
- Saves the trained model to `models/iris_classifier.joblib`

## Requirements
- Python 3.10 or newer
- Internet is only needed to install the Python packages the first time.

## Run on Windows (VS Code)
1. Extract the ZIP file.
2. Open the extracted `DecodeLabs_Project2_Data_Classification` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
5. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   python app.py
   ```
7. Open `http://127.0.0.1:5000` in your browser.

## How it works
1. Loads the labeled Iris dataset provided by scikit-learn.
2. Uses four features: sepal length, sepal width, petal length, and petal width.
3. Splits the data into 80% training and 20% testing records, using stratification to preserve class balance.
4. Scales numeric features and trains KNN on the training split.
5. Evaluates only against the held-out test split.
6. Accepts four new measurements in the web form and predicts one of three Iris species.
7. Saves the fitted pipeline locally in the `models` folder.

## Notes
- Test accuracy can vary if you change the split, dataset, or algorithm.
- A prediction is a model estimate, not a guarantee.
- This project does not collect user accounts or send dataset/model data to a cloud service.
- The Iris dataset is included with scikit-learn, so no external dataset file is required.

## Project files
```text
DecodeLabs_Project2_Data_Classification/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── data/
│   └── README.txt
└── models/
    └── (trained model is generated on first run)
```

## Internship deliverables
For submission, include the source code and screenshots of:
- The running project dashboard
- The dataset/training/testing summary
- The evaluation results
- A sample prediction
