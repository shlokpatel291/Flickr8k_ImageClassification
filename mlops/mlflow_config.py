import mlflow
import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow_tracking(experiment_name):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    logger.info(f"Setting up MLflow tracking for experiment: {experiment_name}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_model_to_mlflow(model, model_name, artifact_path):
    try:
        mlflow.pytorch.log_model(model=model, artifact_path=artifact_path, registered_model_name=model_name)
        logger.info(f"Model '{model_name}' logged to MLflow under path '{artifact_path}'.")
    except Exception as e:
        logger.error(f"Failed to log model '{model_name}' to MLflow: {e}")

def load_model_from_registry(model_name, stage="Production"):
    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Model '{model_name}' loaded successfully from stage '{stage}'.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}' from MLflow registry: {e}")
        raise e
