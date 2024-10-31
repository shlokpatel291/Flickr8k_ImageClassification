import os
import torch
import mlflow
import mlflow.pytorch
from app.model import ImageEncoder, CaptionGenerator
from app.utils import get_flickr8k_dataloader, Vocabulary  # Update function to load Flickr8k
import torch.nn as nn

def train_model(image_dir, captions_file, vocab, batch_size=32, embed_size=256, hidden_size=512, num_layers=1, num_epochs=10):
    mlflow.start_run()

    # Create model instances
    encoder = ImageEncoder(embed_size)
    decoder = CaptionGenerator(embed_size, len(vocab.word2idx), hidden_size, num_layers)

    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    dataloader = get_flickr8k_dataloader(image_dir, captions_file, vocab, batch_size)

    encoder.train()
    decoder.train()
    
    for epoch in range(num_epochs):
        for images, captions in dataloader:
            if images.size(0) == 0:  # Skip empty batches
                continue
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])  # Exclude last token for inputs

            loss = criterion(outputs.view(-1, len(vocab.word2idx)), captions[:, 1:].contiguous().view(-1))  # Exclude start token
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # Log the loss
            mlflow.log_metric("loss", loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save model
    mlflow.pytorch.log_model(encoder, "image_encoder")
    mlflow.pytorch.log_model(decoder, "caption_generator")

    mlflow.end_run()

def load_flickr8k_vocab(captions_file):
    vocab = Vocabulary()
    with open(captions_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split('\t')
            caption = tokens[1].strip()
            vocab.add_sentence(caption)
    return vocab

def run_pipeline(image_dir, captions_file):
    # Load vocabulary
    vocab = load_flickr8k_vocab(captions_file)

    # Create or get experiment
    experiment_name = "Image_Captioning_Experiment"  # Specify your experiment name
    mlflow.set_experiment(experiment_name)

    # Train the model
    train_model(image_dir, captions_file, vocab)


if __name__ == "__main__":
    image_dir = "C:\\Users\\shlok\\Desktop\\Image Caption with MLFlow\\data\\Flickr8k_Dataset"
    captions_file = "C:\\Users\\shlok\\Desktop\\Image Caption with MLFlow\\data\\Flickr8k_text\\Flickr8k.token.txt"
    run_pipeline(image_dir, captions_file)
