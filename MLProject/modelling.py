from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


TARGET = "attrition"


def configure_tracking() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME")
    use_dagshub = os.getenv("USE_DAGSHUB_TRACKING", "").lower() in {"1", "true", "yes"}
    if use_dagshub and owner and repo:
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment("Panji Workflow CI Training")


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if path.is_dir():
        train_path = path / "train.csv"
        test_path = path / "test.csv"
    else:
        train_path = path.parent / "train.csv"
        test_path = path.parent / "test.csv"
    train = pd.read_csv(train_path).drop(columns=["split"], errors="ignore")
    test = pd.read_csv(test_path).drop(columns=["split"], errors="ignore")
    return train.drop(columns=[TARGET]), train[TARGET], test.drop(columns=[TARGET]), test[TARGET]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="employee_attrition_preprocessing")
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--max-depth", default="8")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()
    max_depth = None if str(args.max_depth).lower() == "none" else int(args.max_depth)

    configure_tracking()
    X_train, y_train, X_test, y_test = load_data(Path(args.data_path))
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )

    with mlflow.start_run(run_name="ci-mlflow-project-training") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "model_type": "RandomForestClassifier",
            }
        )
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=X_test.head(5))
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "ci_metrics.json"
            report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="reports")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        (output_dir / "latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        print(f"MLflow run_id={run.info.run_id}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
