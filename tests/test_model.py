import torch
from app.model import ImageEncoder, CaptionGenerator

def test_image_encoder():
    encoder = ImageEncoder(embed_size=256)
    test_image = torch.randn(1, 3, 224, 224)  # Batch of one image

    features = encoder(test_image)

    assert features.shape == (1, 256)

def test_caption_generator():
    vocab_size = 5000
    generator = CaptionGenerator(embed_size=256, vocab_size=vocab_size, hidden_size=512, num_layers=2)

    features = torch.randn(1, 256)  
    captions = torch.randint(0, vocab_size, (1, 10))

    outputs = generator(features, captions)
    assert outputs.shape == (1, 10, vocab_size)
