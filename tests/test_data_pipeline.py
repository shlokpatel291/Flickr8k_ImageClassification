import torch
from app.utils import get_dataloader, Vocabulary

def test_data_pipeline():
    image_dir = "test_images/"
    captions_file = "test_captions.json"
    batch_size = 4

    vocab = Vocabulary()
    vocab.build_from_captions(captions_file)
    dataloader = get_dataloader(image_dir, captions_file, vocab, batch_size)
    data_iter = iter(dataloader)

    images, captions = next(data_iter)

    assert isinstance(images, torch.Tensor)
    assert isinstance(captions, torch.Tensor)
    assert images.shape[0] == batch_size
    assert captions.shape[0] == batch_size
