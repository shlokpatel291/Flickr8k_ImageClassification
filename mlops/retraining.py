from .mlflow_config import load_model_from_registry, log_model_to_mlflow
from app.train import train_model
from app.utils import get_flickr8k_dataloader, Vocabulary
import mlflow

def retrain_model(model_name, image_dir, captions_file, vocab_size, batch_size, num_epochs):
    try:
        current_model = load_model_from_registry(model_name)
        print(f"Loaded model '{model_name}' for retraining.")

        # Build vocabulary and DataLoader
        vocab = Vocabulary()
        vocab.build_from_captions(captions_file)
        dataloader = get_flickr8k_dataloader(image_dir, captions_file, vocab, batch_size)

        # Retrain model
        embed_size = 256
        hidden_size = 512
        num_layers = 2
        retrained_model = train_model(embed_size, vocab_size, hidden_size, num_layers, dataloader, num_epochs)
        print("Model retrained successfully.")

        # Log retrained model to MLflow
        with mlflow.start_run():
            log_model_to_mlflow(retrained_model, model_name, "retrained_caption_generator_model")
            print(f"Retrained model '{model_name}' logged successfully.")

    except Exception as e:
        print(f"Failed to retrain model '{model_name}': {e}")
        raise e
