import torch
import mlflow.pytorch
from .utils import preprocess_image
import io
from PIL import Image

def load_model():
    model_uri = "models:/caption_generator_model/production"
    caption_generator = mlflow.pytorch.load_model(model_uri)
    return caption_generator

def generate_caption(image_bytes):
    # Ensure image_bytes is passed correctly
    image_tensor = preprocess_image(image_bytes)
    
    caption_generator = load_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        features = caption_generator.encode_image(image_tensor)
        caption = caption_generator.generate_caption(features)

    return caption
