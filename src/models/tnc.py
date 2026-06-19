import torch
import torch.nn as nn


class TNCEncoder(nn.Module):
    """Enkoder pojedynczego okna EEG dla TNC.

    Tak jak w CPC, warstwy splotowe odpowiadaja enkoderowi baseline
    (baseline_ae.Autoencoder), aby reprezentacja miala ten sam rozmiar i byla
    porownywalna w protokole linear evaluation. Wejscie: (B, C, T).
    Wyjscie: (B, encoded_size) -- splaszczona reprezentacja okna.
    """

    def __init__(self, num_channels, sequence_length):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, sequence_length)
            out = self.conv(dummy)
            self.encoded_size = out.view(1, -1).size(1)

    def forward(self, x):
        feat = self.conv(x)
        return feat.view(feat.size(0), -1)


class Discriminator(nn.Module):
    """Klasyfikuje pare reprezentacji okien jako sasiednie (1) lub nie (0).

    Wejscie: konkatenacja (z_t, z_other) o wymiarze 2 * encoded_size.
    """

    def __init__(self, encoded_size, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * encoded_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_a, z_b):
        return self.net(torch.cat([z_a, z_b], dim=-1)).squeeze(-1)


class TNCModel(nn.Module):
    """Temporal Neighborhood Coding dla sekwencji okien EEG.

    Dla okna kotwicznego t probkujemy:
      - okna z sasiedztwa czasowego (indeksy w promieniu `neighbor_range`)
        jako przyklady POZYTYWNE,
      - okna spoza sasiedztwa jako przyklady NEGATYWNE.
    Dyskryminator uczy sie odroznic obie grupy, co zmusza enkoder do nadawania
    podobnych reprezentacji oknom bliskim w czasie.

    Uzywamy wariantu PU-learning (Positive-Unlabeled): straty negatywne sa
    wazone wspolczynnikiem `w` poniewaz "odlegle" okna moga byc faktycznie
    podobne (sygnal EEG bywa stacjonarny), wiec traktujemy je jako probki
    nieoznaczone, nie czyste negatywy.

    Po treningu do ewaluacji uzywamy wylacznie enkodera (encode_windows).
    """

    def __init__(self, num_channels, sequence_length, neighbor_range=3,
                 num_samples=5, w=0.05):
        super().__init__()
        self.encoder = TNCEncoder(num_channels, sequence_length)
        self.encoded_size = self.encoder.encoded_size
        self.discriminator = Discriminator(self.encoded_size)
        self.neighbor_range = neighbor_range
        self.num_samples = num_samples
        self.w = w
        self.bce = nn.BCEWithLogitsLoss()

    def encode_windows(self, windows):
        """(num_windows, C, T) -> (num_windows, encoded_size)."""
        return self.encoder(windows)

    def _sample_pairs(self, n, device):
        """Buduje indeksy (anchor, neighbor, non_neighbor) dla sekwencji dlugosci n."""
        anchors, neighbors, non_neighbors = [], [], []
        r = self.neighbor_range

        for t in range(n):
            nb_lo, nb_hi = max(0, t - r), min(n - 1, t + r)
            neighbor_pool = [i for i in range(nb_lo, nb_hi + 1) if i != t]
            non_pool = [i for i in range(n) if i < t - r or i > t + r]
            if not neighbor_pool or not non_pool:
                continue

            # Deterministyczne probkowanie (bez RNG): co k-ty element puli.
            for s in range(self.num_samples):
                nb = neighbor_pool[s % len(neighbor_pool)]
                nn_idx = non_pool[(s * 7 + t) % len(non_pool)]
                anchors.append(t)
                neighbors.append(nb)
                non_neighbors.append(nn_idx)

        if not anchors:
            return None
        return (
            torch.tensor(anchors, device=device),
            torch.tensor(neighbors, device=device),
            torch.tensor(non_neighbors, device=device),
        )

    def forward(self, windows):
        """Liczy strate TNC dla sekwencji okien jednego pacjenta.

        windows: (num_windows, C, T) w kolejnosci czasowej.
        Zwraca: (loss, accuracy) lub (None, 0.0) gdy sekwencja zbyt krotka.
        """
        z = self.encode_windows(windows)            # (N, D)
        n = z.size(0)
        device = z.device

        sampled = self._sample_pairs(n, device)
        if sampled is None:
            return None, 0.0
        a_idx, nb_idx, non_idx = sampled

        z_a = z[a_idx]
        z_nb = z[nb_idx]
        z_non = z[non_idx]

        logits_pos = self.discriminator(z_a, z_nb)
        logits_neg = self.discriminator(z_a, z_non)

        ones = torch.ones_like(logits_pos)
        zeros = torch.zeros_like(logits_neg)

        loss_pos = self.bce(logits_pos, ones)
        # PU-learning: negatywy = nieoznaczone, mieszanka pozytywu (waga w) i negatywu (1-w)
        loss_neg = self.w * self.bce(logits_neg, ones) + (1.0 - self.w) * self.bce(logits_neg, zeros)
        loss = loss_pos + loss_neg

        with torch.no_grad():
            pred_pos = (torch.sigmoid(logits_pos) > 0.5).float()
            pred_neg = (torch.sigmoid(logits_neg) > 0.5).float()
            acc = (pred_pos.sum() + (1 - pred_neg).sum()) / (pred_pos.numel() + pred_neg.numel())
            acc = acc.item()

        return loss, acc
