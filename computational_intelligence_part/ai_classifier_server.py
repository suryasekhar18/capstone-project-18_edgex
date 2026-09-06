from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time  # Import the time library for measuring performance
from pathlib import Path

# Import models and tools from scikit-learn
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# --- Global variables for the trained model ---
model = None
idx_to_class = {}  # To convert model output back to labels like "HIGH"

def train_model():
    """
    Loads data, trains, compares models, and prints all evaluation metrics.
    Runs only once when the server starts.
    """
    global model, idx_to_class

    # 1. Load and prepare the dataset
    try:
        dataset_path = Path(__file__).with_name("priority_dataset.csv")
        df = pd.read_csv(dataset_path)
        X = df[['traffic_density', 'time_of_day', 'weather', 'visibility']].values

        y_labels = df['priority'].values
        classes = list(pd.unique(y_labels))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        y = np.array([class_to_idx[label] for label in y_labels])

    except FileNotFoundError:
        print("FATAL: priority_dataset.csv not found! Please create it.")
        exit()

    # 2. Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("--- Training and Comparing Models on Validation Set ---")

    # --- Model 1: Decision Tree ---
    dt_start_time = time.time()
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_train_time = time.time() - dt_start_time
    dt_acc = accuracy_score(y_val, dt_model.predict(X_val))
    print(f"- Decision Tree Accuracy: {dt_acc:.4f} (Training Time: {dt_train_time:.4f}s)")

    # --- Model 2: Random Forest ---
    rf_start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - rf_start_time
    rf_acc = accuracy_score(y_val, rf_model.predict(X_val))
    print(f"- Random Forest Accuracy: {rf_acc:.4f} (Training Time: {rf_train_time:.4f}s)")

    # --- Model 3: Support Vector Machine (SVM) ---
    svm_start_time = time.time()
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_train_time = time.time() - svm_start_time
    svm_acc = accuracy_score(y_val, svm_model.predict(X_val))
    print(f"- SVM Accuracy        : {svm_acc:.4f} (Training Time: {svm_train_time:.4f}s)")

    # 3. Find and select the best model to use for the server
    models = {
        'Decision Tree': (dt_model, dt_acc),
        'Random Forest': (rf_model, rf_acc),
        'SVM': (svm_model, svm_acc)
    }
    best_model_name = max(models, key=lambda name: models[name][1])
    model = models[best_model_name][0]
    print(f"\n--- Best model selected: {best_model_name} with accuracy {models[best_model_name][1]:.4f} ---")

    # 4. Interpretability Check: Print the rules for the Decision Tree
    print("\n--- Decision Tree Rules (Interpretability Check) ---")
    feature_names = ['traffic_density', 'time_of_day', 'weather', 'visibility']
    tree_rules = export_text(dt_model, feature_names=feature_names)
    print(tree_rules)

# --- API Endpoint ---
@app.route("/predict_priority", methods=["POST"])
def predict_priority():
    """
    Handles incoming requests from the Java simulation to predict priority.
    """
    if model is None:
        return jsonify({"error": "Model is not trained yet"}), 500
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        sample = np.array([
            data["traffic_density"],
            data["time_of_day"],
            data["weather"],
            data["visibility"]
        ])
        sample_reshaped = sample.reshape(1, -1)

        # Measure prediction latency
        pred_start_time = time.time()
        pred_idx = model.predict(sample_reshaped)[0]
        pred_latency_ms = (time.time() - pred_start_time) * 1000  # in milliseconds

        # Print latency for verification
        print(f"Prediction Latency: {pred_latency_ms:.4f} ms")

        priority = idx_to_class[pred_idx]
        return jsonify({"priority": priority})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Run Server ---
if __name__ == "__main__":
    train_model()
    print("--- AI Classifier Server Ready ---")
    app.run(host="0.0.0.0", port=5000)