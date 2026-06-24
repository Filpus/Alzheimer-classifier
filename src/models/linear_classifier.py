import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, encoder, encoded_size):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder

        if self.encoder is not None:  # <-- Dodany warunek dla kompatybilności
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(encoded_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def train(self, mode=True):
        # Trenujemy TYLKO glowice. Enkoder ma zostac w trybie eval, inaczej jego
        # warstwy BatchNorm uzywalyby statystyk batcha z linear-probe i nadpisywaly
        # running_mean/var nauczone podczas SSL (requires_grad=False tego NIE blokuje
        # -- robi to dopiero tryb eval). To dawalo rozjazd cech train vs eval i
        # zanizalo AUC. Po tej zmianie enkoder jest zamrozony tez statystycznie.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        if self.encoder is not None:  # <-- Omija enkoder, jeśli podano None
            with torch.no_grad():
                x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x