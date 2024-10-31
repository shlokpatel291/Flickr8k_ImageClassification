import os
import torch
import mlflow
import mlflow.pytorch
from torchvision import transforms
from app.model import ImageEncoder, CaptionGenerator
from app.utils import get_flickr8k_dataloader, Vocabulary  # Adjusted for Flickr8k
import torch.nn as nn

def train_model(image_dir, captions_file, vocab, batch_size=32, embed_size=256, hidden_size=512, num_layers=1, num_epochs=20):
    mlflow.start_run()

    # Create model instances
    encoder = ImageEncoder(embed_size)
    decoder = CaptionGenerator(embed_size, len(vocab.word2idx), hidden_size, num_layers)

    # Loss function (ignores <pad> token with index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    dataloader = get_flickr8k_dataloader(image_dir, captions_file, vocab, batch_size)

    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        for images, captions in dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])  # Exclude last token for inputs

            targets = captions[:, 1:].contiguous().view(-1)  # Exclude <start> token from targets
            outputs = outputs.view(-1, len(vocab.word2idx))

            loss = criterion(outputs, targets)
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            mlflow.log_metric("loss", loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save models
    mlflow.pytorch.log_model(encoder, "image_encoder")
    mlflow.pytorch.log_model(decoder, "caption_generator")

    mlflow.end_run()

def load_flickr8k_vocab(captions_file):
    vocab = Vocabulary()
    with open(captions_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split('\t')
            caption = tokens[1].strip()
            vocab.add_sentence(f"<start> {caption} <end>")  # Add special tokens
    return vocab

def run_pipeline(image_dir, captions_file):
    vocab = load_flickr8k_vocab(captions_file)

    experiment_name = "Image_Captioning_Experiment"
    mlflow.set_experiment(experiment_name)

    train_model(image_dir, captions_file, vocab)

if __name__ == "__main__":
    image_dir = "C:\\Users\\shlok\\Desktop\\Image Caption with MLFlow\\data\\Flickr8k_Dataset"
    captions_file = "C:\\Users\\shlok\\Desktop\\Image Caption with MLFlow\\data\\Flickr8k_text\\Flickr8k.token.txt"
    run_pipeline(image_dir, captions_file)
