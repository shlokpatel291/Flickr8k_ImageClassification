from .mlflow_config import setup_mlflow_tracking, log_model_to_mlflow
from app.train import train_model
from app.utils import get_flickr8k_dataloader, Vocabulary
import mlflow

def run_pipeline(experiment_name, model_name, image_dir, captions_file, vocab_size, batch_size, num_epochs):
    try:
        setup_mlflow_tracking(experiment_name)
        print(f"MLflow tracking set up for experiment: {experiment_name}")

        # Build vocabulary from captions
        vocab = Vocabulary()
        vocab.build_from_captions(captions_file)
        print("Vocabulary built successfully.")

        # Create DataLoader for the dataset
        dataloader = get_flickr8k_dataloader(image_dir, captions_file, vocab, batch_size)
        print("DataLoader created successfully.")

        # Model parameters
        embed_size = 256
        hidden_size = 512
        num_layers = 2

        # Train the model
        trained_model = train_model(embed_size, vocab_size, hidden_size, num_layers, dataloader, num_epochs)
        print("Model trained successfully.")

        # Log the model to MLflow
        with mlflow.start_run():
            log_model_to_mlflow(trained_model, model_name, "caption_generator_model")

        print(f"Pipeline completed successfully for experiment '{experiment_name}'.")
    
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
        raise e
