"""Wspolna logika treningu samonadzorowanych reprezentacji (CPC/TNC) i klasyfikacji liniowej.

Jedno zrodlo prawdy dla skryptow CLI (train_cpc.py, train_tnc.py) oraz notebooka
raportowego. Skrypty wykonuja pelna CV i zapisuja enkodery; notebook dokłada wokol
tego warstwe cache wynikow i wariantow 'best'.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from src.models.linear_classifier import LinearClassifier


def train_ssl(model, seq_loader, epochs, device, lr=1e-3, verbose=True, loss_name="loss"):
    """Trenuje model SSL (CPC/TNC) na sekwencjach per-pacjent.

    model.forward(seq) musi zwracac (loss, acc_zadania_pretekstowego) lub (None, 0.0)
    gdy sekwencja jest za krotka. Zwraca historie sredniej straty na epoke.
    """
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        model.train()
        tot, acc, n = 0.0, 0.0, 0
        for seq, _ in seq_loader:
            seq = seq.to(device)
            opt.zero_grad()
            loss, a = model(seq)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            tot += loss.item(); acc += a; n += 1
        hist.append(tot / max(n, 1))
        if verbose:
            print(f"  epoka {ep+1:2d}/{epochs} | {loss_name} {tot/max(n,1):.4f} | acc_pretext {acc/max(n,1):.3f}")
    return hist


def linear_probe(encoder, encoded_size, train_loader, val_loader, epochs, device, lr=5e-3):
    """Klasyfikacja liniowa: zamrozony enkoder + plytka glowica trenowana na cechach.

    Zwraca najlepsze (accuracy, roc_auc) na zbiorze walidacyjnym po `epochs` epokach.
    """
    encoder.eval()

    # --- [MINIMALNA MODYFIKACJA]: Ekstrakcja cech przed pętlą treningową ---
    def get_features(loader):
        feats, targets = [], []
        with torch.no_grad():
            for X, y in loader:
                out = encoder(X.to(device))
                feats.append(out.view(out.size(0), -1).cpu())
                targets.append(y)
        return DataLoader(TensorDataset(torch.cat(feats), torch.cat(targets)), batch_size=loader.batch_size,
                          shuffle=(loader == train_loader))

    fast_train_loader = get_features(train_loader)
    fast_val_loader = get_features(val_loader)
    # ----------------------------------------------------------------------

    clf = LinearClassifier(None, encoded_size).to(device)  # encoder=None, bo cechy są już wyciągnięte
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(clf.classifier.parameters(), lr=lr)
    best_acc, best_auc = 0.0, 0.0

    for _ in range(epochs):
        clf.train()
        for X, y in fast_train_loader:  # Zmiana na fast_train_loader
            X = X.to(device)
            y = y.float().to(device).unsqueeze(1)
            opt.zero_grad()
            crit(clf(X), y).backward()
            opt.step()
        clf.eval()
        probs, targets = [], []
        with torch.no_grad():
            for X, y in fast_val_loader:  # Zmiana na fast_val_loader
                probs.extend(torch.sigmoid(clf(X.to(device))).cpu().numpy())
                targets.extend(y.numpy())
        probs = np.array(probs).ravel()
        targets = np.array(targets).ravel()
        best_acc = max(best_acc, accuracy_score(targets, (probs > 0.5)))
        best_auc = max(best_auc, roc_auc_score(targets, probs))
    return best_acc, best_auc
