import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
if __name__ == "__main__":
    # Data
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Logistic Regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Make predictions
    predictions = lr.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Start MLflow tracking
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("solver", lr.solver)
        mlflow.log_param("C", lr.C)
        mlflow.log_param("intercept", lr.intercept_[0])
        
        # Log model metrics (accuracy)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        signature = infer_signature(X_test, predictions)
        mlflow.sklearn.log_model(lr, "logistic-regression-model", signature=signature)

        # Output results
        print(f"Model Accuracy: {accuracy}")
        print(f"Model saved in run {mlflow.active_run().info.run_uuid}")