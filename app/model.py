import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        # Load a pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)

        # Freeze ResNet parameters to avoid backpropagation through it
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove the fully connected layer of ResNet
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # Linear layer to project ResNet output to the embedding size
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()  # Optional: can remove if not needed

    def forward(self, images):
        # Extract feature maps
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # Flatten features
        # Project features to the desired embedding size
        features = self.embed(features)
        return features


class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(CaptionGenerator, self).__init__()
        
        # Embedding layer to convert word indices to embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer to project LSTM outputs to the vocabulary size
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Convert captions to embeddings
        embeddings = self.embed(captions)
        
        # Concatenate image features with the caption embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(inputs)
        
        # Project LSTM outputs to vocabulary size
        outputs = self.linear(lstm_out)
        return outputs

    def generate_caption(self, features, vocab, max_length=20):
        """Generate a caption for an image feature using greedy decoding."""
        outputs = []
        input = features.unsqueeze(1)  # Initial input is the image features
        states = None

        for _ in range(max_length):
            # Pass through LSTM
            lstm_out, states = self.lstm(input, states)
            output = self.linear(lstm_out.squeeze(1))  # Project to vocab size
            
            # Get the word with the max probability
            _, predicted = output.max(1)  
            predicted_word = vocab.idx2word[predicted.item()]  # Convert index to word
            
            outputs.append(predicted_word)
            input = self.embed(predicted).unsqueeze(1)  # Prepare the next input
            
            if predicted_word == '<end>':  # Stop if the <end> token is generated
                break

        return ' '.join(outputs)
