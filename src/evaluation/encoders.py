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
from src.models.mamba_model import MambaEncoder


# nazwa modelu -> (podfolder, prefiks pliku) z wagami enkodera.
# Osobne podfoldery uniemozliwiaja nakladanie sie outputow DVC miedzy
# etapami train_repr (AE) / train_repr_cpc / train_repr_tnc.
ENCODER_LOCATION = {
    "ae": ("models/representations/ae", "encoder"),
    "cpc": ("models/representations/cpc", "encoder"),
    "tnc": ("models/representations/tnc", "encoder"),
    "mamba": ("models/representations/mamba", "encoder"),
}

SUPPORTED_MODELS = tuple(ENCODER_LOCATION.keys())


def encoder_dir(model_name):
    return ENCODER_LOCATION[model_name][0]


def encoder_path(model_name, fold):
    folder, prefix = ENCODER_LOCATION[model_name]
    return f"{folder}/{prefix}_fold_{fold}.pth"


def mamba_dmodel_from_ckpt(state_dict):
    """Odczytuje d_model z zapisanego MambaEncoder (embed.patch.weight ma ksztalt [d_model, C, patch]).

    Rozne warianty Mamby (mamba d_model=64, mamba_best np. 128) trenowano z roznym d_model,
    a build_encoder domyslnie tworzy d_model=32 -> stad odczytujemy d_model wprost z wag,
    by zbudowac enkoder o zgodnej architekturze (inaczej load_state_dict rzuca size mismatch).
    """
    return int(state_dict["embed.patch.weight"].shape[0])


def build_encoder(model_name, num_channels, sequence_length, device, ckpt_state=None):
    """Tworzy enkoder danego modelu (bez zaladowanych wag) i zwraca (encoder, encoded_size).

    Dla Mamby, gdy podano ckpt_state (state_dict zapisanego enkodera), d_model jest odczytany
    z wag -- enkoder ewaluacyjny ma wtedy te sama architekture co wytrenowany (rozne warianty
    Mamby maja rozne d_model). Pozostale modele maja jeden, staly wymiar reprezentacji.
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
    if model_name == "mamba":
        d_model = mamba_dmodel_from_ckpt(ckpt_state) if ckpt_state is not None else 64
        enc = MambaEncoder(num_channels, sequence_length, d_model=d_model).to(device)
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
