import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, num_channels, sequence_length):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, sequence_length)
            dummy_output = self.encoder(dummy_input)
            self.encoded_size = dummy_output.view(1, -1).size(1)
            
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(16, num_channels, kernel_size=15, stride=1, padding=7)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded