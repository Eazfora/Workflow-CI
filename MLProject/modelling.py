import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
recall_score, f1_score, roc_auc_score,
confusion_matrix, roc_curve)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",      type=int, default=200)
parser.add_argument("--max_depth",         type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=5)
parser.add_argument("--min_samples_leaf",  type=int, default=2)
args = parser.parse_args()

DAGSHUB_USERNAME  = os.getenv("firmansyahagung239", "Agung-Firmansyah")
DAGSHUB_REPO_NAME = os.getenv("Eksperimen_SML_Agung-Firmansyah", "student-performance-ml")

DATA_DIR   = os.path.join(os.path.dirname(__file__),
"student_performance_preprocessing")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
TARGET_COL = "passed"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_true, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Tidak Lulus", "Lulus"],
                yticklabels=["Tidak Lulus", "Lulus"])
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve(y_true, y_prob, path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
    label=f"AUC={roc_auc_score(y_true, y_prob):.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_feature_importance(model, feature_names, path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(indices)), importances[indices][::-1], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    X_train, X_test, y_train, y_test = load_data()

    params = {
        "n_estimators"     : args.n_estimators,
        "max_depth"        : args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf" : args.min_samples_leaf,
        "random_state"     : 42,
        "n_jobs"           : -1,
    }

    with mlflow.start_run(run_name="CI_RandomForest_Run"):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv     = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        metrics = {
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision" : precision_score(y_test, y_pred),
            "recall" : recall_score(y_test, y_pred),
            "f1_score" : f1_score(y_test, y_pred),
            "roc_auc" : roc_auc_score(y_test, y_prob),
            "cv_mean_accuracy": cv.mean(),
            "cv_std_accuracy" : cv.std(),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="StudentPerformanceClassifier_CI"
        )

        # Artefak tambahan
        cm_path  = save_confusion_matrix(y_test, y_pred)
        roc_path = save_roc_curve(y_test, y_prob)
        fi_path  = save_feature_importance(model, list(X_train.columns))

        mlflow.log_artifact(cm_path,  "plots")
        mlflow.log_artifact(roc_path, "plots")
        mlflow.log_artifact(fi_path,  "plots")

        print(f"[CI] Run selesai. Run ID: {mlflow.active_run().info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
