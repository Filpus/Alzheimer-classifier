"""Wspolna logika treningu samonadzorowanych reprezentacji (CPC/TNC) i klasyfikacji liniowej.

Jedno zrodlo prawdy dla skryptow CLI (train_cpc.py, train_tnc.py) oraz notebooka
raportowego. Skrypty wykonuja pelna CV i zapisuja enkodery; notebook dokłada wokol
tego warstwe cache wynikow i wariantow 'best'.
"""

import copy

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


def _eval_head_on_loader(clf, val_loader, device):
    """Inference: zwraca (acc, auc) glowicy `clf` na `val_loader` (per okno)."""
    clf.eval()
    probs, targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            probs.extend(torch.sigmoid(clf(X.to(device))).cpu().numpy())
            targets.extend(y.numpy())
    probs = np.array(probs).ravel(); targets = np.array(targets).ravel()
    return accuracy_score(targets, (probs > 0.5)), roc_auc_score(targets, probs)


def eval_head(encoder, encoded_size, head_state, val_loader, device):
    """Bez treningu: wczytuje zapisana glowice (state_dict modulu .classifier) na zamrozony
    enkoder i liczy (acc, auc) per okno. Uzywane, gdy glowica jest juz na dysku."""
    clf = LinearClassifier(encoder, encoded_size).to(device)
    clf.classifier.load_state_dict(head_state)
    return _eval_head_on_loader(clf, val_loader, device)


def linear_probe(encoder, encoded_size, train_loader, val_loader, epochs, device,
                 lr=5e-3, patience=None):
    """Klasyfikacja liniowa: zamrozony enkoder + plytka glowica trenowana na cechach.

    Mierzy (acc, auc) per okno po KAZDEJ epoce i sledzi NAJLEPSZA epoke wg AUC. Gdy `patience`
    jest podane, stosuje early-stopping: przerywa, gdy AUC nie poprawi sie przez `patience` epok.

    Zwraca (best_acc, best_auc, best_head_state), gdzie best_head_state to deepcopy state_dict
    modulu .classifier z epoki najlepszego AUC (do zapisu w models/evaluation/ i pozniejszego
    odtworzenia metryk bez treningu).
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
    # MERGE (damian+mamba): zachowany interfejs 3-wartosciowy (best_head_state) + early-stopping (patience)
    # z brancha damian; optymalizacja feature-caching (fast_* loadery, encoder=None) z brancha mamba.
    best_head_state = copy.deepcopy(clf.classifier.state_dict())
    no_improve = 0
    for _ in range(epochs):
        clf.train()
        for X, y in fast_train_loader:  # cechy wyciagniete raz przed petla (optymalizacja mamby)
            X = X.to(device); y = y.float().to(device).unsqueeze(1)
            opt.zero_grad(); crit(clf(X), y).backward(); opt.step()
        acc, auc = _eval_head_on_loader(clf, fast_val_loader, device)
        best_acc = max(best_acc, acc)
        if auc > best_auc:
            best_auc = auc
            best_head_state = copy.deepcopy(clf.classifier.state_dict())  # glowica z epoki max AUC
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                break
    return best_acc, best_auc, best_head_state
