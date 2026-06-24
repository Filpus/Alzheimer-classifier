import torch
import torch.nn as nn

from src.models.encoder import WindowEncoder, ProjectionHead


class CPCEncoder(WindowEncoder):
    """Enkoder okna EEG (g_enc) dla CPC -- wspolny WindowEncoder (128-d, global pooling).

    Identyczny z TNCEncoder, dzieki czemu porownanie CPC vs TNC ocenia samą metodę
    SSL, a nie rozne architektury. Wejscie (B, C, T) -> wyjscie (B, 128).
    """
    pass


class CPCModel(nn.Module):
    """Contrastive Predictive Coding dla sekwencji okien EEG.

    Sekwencja okien jednego pacjenta jest kodowana per okno przez g_enc do
    wektora z_t (splaszczona mapa cech). Autoregresor GRU (g_ar) buduje
    kontekst c_t podsumowujacy przeszlosc. Dla kazdego kroku predykcji k
    osobna macierz W_k mapuje c_t na przewidywany z_{t+k}; trening odbywa sie
    strata InfoNCE (zgadnij prawdziwe przyszle okno sposrod negatywow z batcha).

    Po treningu do ewaluacji uzywamy wylacznie g_enc (encode_windows),
    dokladnie jak enkodera baseline.
    """

    def __init__(self, num_channels, sequence_length, context_dim=128, prediction_steps=4,
                 use_projection=False, projection_dim=64):
        super().__init__()
        self.encoder = CPCEncoder(num_channels, sequence_length)
        self.encoded_size = self.encoder.encoded_size
        self.prediction_steps = prediction_steps
        self.context_dim = context_dim

        # Opcjonalna glowica projekcyjna (SimCLR): strata InfoNCE liczona na g(z), nie na z.
        # Do EWALUACJI uzywamy surowego enkodera (encode_windows) -- glowica jest odrzucana.
        self.use_projection = use_projection
        self.projection = ProjectionHead(self.encoded_size, self.encoded_size, projection_dim) if use_projection else None
        # wymiar przestrzeni, w ktorej liczona jest strata (z lub g(z))
        self._space_dim = projection_dim if use_projection else self.encoded_size

        # Autoregresor podsumowujacy historie reprezentacji
        self.gru = nn.GRU(
            input_size=self._space_dim,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
        )

        # Osobna projekcja liniowa W_k dla kazdego horyzontu predykcji
        self.predictors = nn.ModuleList(
            [nn.Linear(context_dim, self._space_dim) for _ in range(prediction_steps)]
        )

    def encode_windows(self, windows):
        """(num_windows, C, T) -> (num_windows, encoded_size). Reprezentacje per okno.

        Zawsze surowy enkoder (128-d) -- to jest reprezentacja uzywana w klasyfikacji liniowej,
        niezaleznie od tego, czy w treningu uzyto glowicy projekcyjnej.
        """
        feat = self.encoder(windows)
        return feat.view(feat.size(0), -1)

    def forward(self, windows):
        """Liczy strate InfoNCE dla sekwencji okien jednego pacjenta.

        windows: (num_windows, num_channels, window_samples) w kolejnosci czasowej.
        Zwraca: (loss, accuracy) usrednione po krokach predykcji.
        """
        z = self.encode_windows(windows)            # (N, encoded_size) -- surowa reprezentacja
        if self.use_projection:
            z = self.projection(z)                  # (N, projection_dim) -- strata liczona na projekcji
        n = z.size(0)
        device = z.device

        # Sekwencja dla GRU: (batch=1, N, D) -> konteksty c_t: (N, context_dim)
        c, _ = self.gru(z.unsqueeze(0))
        c = c.squeeze(0)

        total_loss = z.new_zeros(())
        total_acc = 0.0
        valid_steps = 0

        for k in range(1, self.prediction_steps + 1):
            # Kontekst w czasie t przewiduje z_{t+k}; t = 0 .. n-1-k
            if n - k <= 1:
                # Za malo par dla tego horyzontu (potrzeba >=2 negatywow/pozytywow)
                continue

            c_t = c[: n - k]                          # (M, context_dim)
            z_target = z[k:]                          # (M, D)
            pred = self.predictors[k - 1](c_t)        # (M, D)

            # Macierz podobienstwa: wiersz i (predykcja) vs kolumna j (cel)
            # log-bilinear scoring f_k(z_{t+k}, c_t) = z^T W_k c_t == pred . z
            logits = pred @ z_target.t()              # (M, M)
            targets = torch.arange(logits.size(0), device=device)

            loss_k = nn.functional.cross_entropy(logits, targets)
            total_loss = total_loss + loss_k

            with torch.no_grad():
                acc_k = (logits.argmax(dim=1) == targets).float().mean().item()
            total_acc += acc_k
            valid_steps += 1

        if valid_steps == 0:
            # Pacjent z bardzo krotka sekwencja -- brak gradientu, pomijany w treningu
            return None, 0.0

        return total_loss / valid_steps, total_acc / valid_steps
