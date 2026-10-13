"""
DecodeLabs Project 2: Data Classification Using AI
A small Flask web app for training, evaluating, and testing a supervised classifier.
Run: python app.py
"""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iris_classifier.joblib"
app = Flask(__name__)

def load_dataset():
    """Load the small Iris dataset bundled with scikit-learn."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, list(iris.feature_names), list(iris.target_names)

def train_model():
    df, features, class_names = load_dataset()
    X = df[features]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=class_names)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "class_names": class_names}, MODEL_PATH)
    return {
        "model": model, "features": features, "class_names": class_names,
        "accuracy": accuracy, "report": report, "matrix": matrix.tolist(),
        "test_size": len(y_test), "train_size": len(y_train),
        "dataset_size": len(df), "df": df
    }

RESULTS = train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {"sepal length (cm)": "5.1", "sepal width (cm)": "3.5",
              "petal length (cm)": "1.4", "petal width (cm)": "0.2"}
    if request.method == "POST":
        try:
            values = {feature: request.form.get(feature, "").strip() for feature in RESULTS["features"]}
            row = []
            for feature in RESULTS["features"]:
                value = float(values[feature])
                if value <= 0:
                    raise ValueError("All measurements must be greater than zero.")
                row.append(value)
            label = RESULTS["model"].predict(pd.DataFrame([row], columns=RESULTS["features"]))[0]
            prediction = label
        except (TypeError, ValueError) as exc:
            error = str(exc) or "Please enter valid numeric measurements."
    return render_template(
        "index.html", features=RESULTS["features"], class_names=RESULTS["class_names"],
        accuracy=RESULTS["accuracy"], train_size=RESULTS["train_size"],
        test_size=RESULTS["test_size"], dataset_size=RESULTS["dataset_size"],
        matrix=RESULTS["matrix"], report=RESULTS["report"], prediction=prediction,
        error=error, values=values
    )

if __name__ == "__main__":
    app.run(debug=True)
