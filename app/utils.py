import os
import torch
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.index = 4

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2idx:
                self.word2idx[word] = self.index
                self.idx2word[self.index] = word
                self.index += 1

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, captions_file, vocab):
        self.image_dir = image_dir
        self.vocab = vocab
        self.captions = self.load_captions(captions_file)

    def load_captions(self, captions_file):
        captions = []
        with open(captions_file, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                image_id, caption = tokens[0].split('#')[0], tokens[1]
                captions.append((image_id, caption))
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image_id, caption = self.captions[index]
        image_path = os.path.join(self.image_dir, image_id)

        # Check if the image exists
        if not os.path.isfile(image_path):
            print(f"Image not found: {image_path}")
            return None, None  # Return None for missing image

        image = preprocess_image(image_path)  # Use the preprocess_image function
        caption = [self.vocab.word2idx.get(word, self.vocab.word2idx["<unk>"]) for word in caption.split()]
        return image, caption

def get_flickr8k_dataloader(image_dir, captions_file, vocab, batch_size):
    dataset = Flickr8kDataset(image_dir, captions_file, vocab)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

def collate_fn(batch):
    # Filter out None values (missing images)
    batch = [item for item in batch if item[0] is not None]

    if not batch:  # If the batch is empty after filtering, return empty tensors
        return torch.empty(0, 3, 224, 224), torch.empty(0)  # Shape for images and captions

    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = [torch.tensor(c) for c in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

def load_image(image_path):
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return None  # Return None if the image does not exist

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)