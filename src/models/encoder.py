import torch.nn as nn


class WindowEncoder(nn.Module):
    """Wspolny enkoder pojedynczego okna EEG dla CPC i TNC.

    Glebszy niz baseline AE (4 bloki splotowe 32->64->128->128) zakonczony
    global average poolingiem -> zwarta reprezentacja 128-wymiarowa zamiast
    splaszczonej mapy 12288-d. Dzieki temu:
      * glowica linear-eval ma ~kilka tys. parametrow zamiast ~393k (mniej przeucza),
      * reprezentacja jest niezalezna od dlugosci okna (AdaptiveAvgPool),
      * enkoder jest IDENTYCZNY dla CPC i TNC -> porownanie tych metod pozostaje uczciwe.

    Wejscie:  (B, num_channels, window_samples)
    Wyjscie:  (B, 128)
    """

    def __init__(self, num_channels, sequence_length=None):
        super().__init__()
        # sequence_length nieuzywane (AdaptiveAvgPool ustala wymiar wyjscia) -
        # zachowane w sygnaturze dla zgodnosci z dotychczasowymi wywolaniami.
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.encoded_size = 128

    def forward(self, x):
        return self.conv(x).flatten(1)   # (B, 128)


class ProjectionHead(nn.Module):
    """Glowica projekcyjna (za SimCLR, Chen i in. 2020) uzywana TYLKO podczas treningu SSL.

    Strata kontrastywna liczona jest na projekcji g(z), nie wprost na reprezentacji z.
    Do ewaluacji (klasyfikacja liniowa) glowice ODRZUCAMY i uzywamy samego enkodera (128-d) --
    SimCLR pokazal, ze taka posrednia projekcja poprawia jakosc samej reprezentacji.

    Wspolna dla CPC i TNC -> porownanie metod pozostaje uczciwe (ta sama architektura).
    Wejscie (B, in_dim) -> wyjscie (B, out_dim).
    """

    def __init__(self, in_dim=128, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)
