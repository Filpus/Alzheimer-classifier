"""Sciezki oraz zapis/odczyt artefaktow ewaluacji reprezentacji.

Dwa rodzaje artefaktow, oba adresowane po WARIANCIE (nie po model_name jak
src/evaluation/encoders.py), bo warianty 'best' (tnc_best, cpc_best) maja wlasne
podkatalogi nieznane encoders.py:

  * wagi enkodera   -> models/representations/<wariant>/encoder_fold_<f>.pth   (-> DVC)
  * glowice liniowe -> models/evaluation/<wariant>/head_<level>_fold_<f>.pth   (-> DVC)

`level` glowicy: 'win' (epoka max AUC per-okno) lub 'pat' (epoka max AUC per-pacjent).
Wszystkie funkcje przyjmuja `root` (katalog projektu) jako argument - brak stanu globalnego.
"""

import os

import torch


# ---------- wagi enkoderow ----------
def encoder_path(root, variant, fold):
    return os.path.join(root, 'models', 'representations', variant, f'encoder_fold_{fold}.pth')


def save_encoder(model, root, variant, fold):
    """Zapisuje state_dict enkodera modelu SSL/AE. Zwraca sciezke."""
    path = encoder_path(root, variant, fold)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.encoder.state_dict(), path)
    return path


def all_encoders_exist(root, variant, n_splits):
    return all(os.path.exists(encoder_path(root, variant, f)) for f in range(n_splits))


# ---------- glowice liniowe ----------
def head_path(root, variant, fold, level):
    return os.path.join(root, 'models', 'evaluation', variant, f'head_{level}_fold_{fold}.pth')


def save_head(head_state, root, variant, fold, level):
    """Zapisuje state_dict glowicy (modul LinearClassifier.classifier). Zwraca sciezke."""
    path = head_path(root, variant, fold, level)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(head_state, path)
    return path


def load_head(root, variant, fold, level, device):
    return torch.load(head_path(root, variant, fold, level), map_location=device)


def all_heads_exist(root, variant, n_splits, level):
    return all(os.path.exists(head_path(root, variant, f, level)) for f in range(n_splits))
