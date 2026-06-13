import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, encoder, encoded_size):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(encoded_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x