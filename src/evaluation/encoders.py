"""Wspolny dostep do enkoderow wszystkich modeli reprezentacji.

Cel: protokol linear evaluation (RF / klasyfikator liniowy) ma byc identyczny
dla baseline (AE), CPC i TNC. Kazdy z tych modeli zapisuje enkoder jako
state_dict pod ustalona nazwa pliku. Ten modul buduje odpowiedni enkoder,
laduje wagi i udostepnia jednolita metode ekstrakcji cech per okno.
"""

import torch
import torch.nn as nn

from src.models.baseline_ae import Autoencoder
from src.models.cpc import CPCEncoder
from src.models.tnc import TNCEncoder


# nazwa modelu -> (podfolder, prefiks pliku) z wagami enkodera.
# Osobne podfoldery uniemozliwiaja nakladanie sie outputow DVC miedzy
# etapami train_repr (AE) / train_repr_cpc / train_repr_tnc.
ENCODER_LOCATION = {
    "ae": ("models/representations/ae", "encoder"),
    "cpc": ("models/representations/cpc", "encoder"),
    "tnc": ("models/representations/tnc", "encoder"),
}

SUPPORTED_MODELS = tuple(ENCODER_LOCATION.keys())


def encoder_dir(model_name):
    return ENCODER_LOCATION[model_name][0]


def encoder_path(model_name, fold):
    folder, prefix = ENCODER_LOCATION[model_name]
    return f"{folder}/{prefix}_fold_{fold}.pth"


def build_encoder(model_name, num_channels, sequence_length, device):
    """Tworzy enkoder danego modelu (bez zaladowanych wag) i zwraca (encoder, encoded_size).

    Wszystkie enkodery dziela te sama architekture splotowa, wiec encoded_size
    jest identyczny -- to gwarantuje porownywalnosc reprezentacji.
    """
    if model_name == "ae":
        ae = Autoencoder(num_channels, sequence_length).to(device)
        return ae.encoder, ae.encoded_size
    if model_name == "cpc":
        enc = CPCEncoder(num_channels, sequence_length).to(device)
        return enc, enc.encoded_size
    if model_name == "tnc":
        enc = TNCEncoder(num_channels, sequence_length).to(device)
        return enc, enc.encoded_size
    raise ValueError(f"Nieznany model '{model_name}'. Dostepne: {SUPPORTED_MODELS}")


def load_encoder(model_name, fold, num_channels, sequence_length, device):
    """Buduje enkoder, laduje wagi z pliku foldu i ustawia tryb eval."""
    encoder, encoded_size = build_encoder(model_name, num_channels, sequence_length, device)
    path = encoder_path(model_name, fold)
    encoder.load_state_dict(torch.load(path, map_location=device))
    encoder.eval()
    return encoder, encoded_size


def extract_features(encoder, X_batch):
    """(B, C, T) -> (B, encoded_size) splaszczona reprezentacja okna.

    Jednolita dla wszystkich modeli: CPCEncoder/Autoencoder.encoder zwracaja
    mape (B, 32, L'), TNCEncoder zwraca juz (B, encoded_size); flatten ujednolica.
    """
    feat = encoder(X_batch)
    return feat.view(feat.size(0), -1)
