import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub

# Initialize DAGsHub
dagshub.init(repo_owner='your_repo_owner', repo_name='your_repo_name', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/your_repo_owner/your_repo_name.mlflow")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Define evaluation metrics function
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return accuracy, precision, recall, f1, roc_auc

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load Sonar dataset
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    try:
        data = pd.read_csv(csv_url, header=None)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split into features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert target to binary labels (Mine: 0, Rock: 1)
    y = y.apply(lambda x: 1 if x == "R" else 0)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameters
    penalty = sys.argv[1] if len(sys.argv) > 1 else 'l2'
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    solver = sys.argv[3] if len(sys.argv) > 3 else 'lbfgs'

    with mlflow.start_run():
        # Logistic Regression model
        lr = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42)
        lr.fit(X_train, y_train)

        # Predict on test data
        y_pred = lr.predict(X_test)

        # Evaluate model
        accuracy, precision, recall, f1, roc_auc = eval_metrics(y_test, y_pred)

        print(f"Logistic Regression model (penalty={penalty}, C={C}, solver={solver}):")
        print(f"  Accuracy: {accuracy}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  F1 Score: {f1}")
        print(f"  ROC AUC: {roc_auc}")

        # Log parameters and metrics
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        signature = infer_signature(X_train, lr.predict(X_train))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Register model if using remote server
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="LogisticRegressionSonarModel", signature=signature)
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
