import torch
import torch.nn as nn

from src.models.encoder import WindowEncoder, ProjectionHead


class TNCEncoder(WindowEncoder):
    """Enkoder okna EEG dla TNC -- wspolny WindowEncoder (128-d, global pooling).

    Identyczny z CPCEncoder, dzieki czemu porownanie TNC vs CPC ocenia samą metodę
    SSL, a nie rozne architektury. Wejscie (B, C, T) -> wyjscie (B, 128).
    """
    pass


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
                 num_samples=5, w=0.05, use_projection=False, projection_dim=64):
        super().__init__()
        self.encoder = TNCEncoder(num_channels, sequence_length)
        self.encoded_size = self.encoder.encoded_size

        # Opcjonalna glowica projekcyjna (SimCLR): dyskryminator dziala na g(z), nie na z.
        # Do EWALUACJI uzywamy surowego enkodera (encode_windows) -- glowica jest odrzucana.
        self.use_projection = use_projection
        self.projection = ProjectionHead(self.encoded_size, self.encoded_size, projection_dim) if use_projection else None
        disc_dim = projection_dim if use_projection else self.encoded_size

        self.discriminator = Discriminator(disc_dim)
        self.neighbor_range = neighbor_range
        self.num_samples = num_samples
        self.w = w
        self.bce = nn.BCEWithLogitsLoss()

    def encode_windows(self, windows):
        """(num_windows, C, T) -> (num_windows, encoded_size). Zawsze surowy enkoder
        -- reprezentacja uzywana w klasyfikacji liniowej (glowica projekcyjna pomijana)."""
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
        z = self.encode_windows(windows)            # (N, encoded_size) -- surowa reprezentacja
        if self.use_projection:
            z = self.projection(z)                  # (N, projection_dim) -- dyskryminator na projekcji
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
