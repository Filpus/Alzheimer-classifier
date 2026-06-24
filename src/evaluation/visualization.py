"""Wizualizacje jakosci reprezentacji - inference z zapisanych enkoderow i glowic (bez treningu).

  pca_features_2d / plot_pca_grid - rzut PCA 2D cech enkodera (per okno, zbiory walidacyjne
      wszystkich foldow, enkoder swojego foldu -> brak leakage). Pokazuje LINIOWA separowalnosc
      reprezentacji - spojne z protokolem linear evaluation.
  patient_roc / plot_patient_roc - krzywe ROC na poziomie PACJENTA, predykcje 'pooled' ze wszystkich
      foldow (kazdy pacjent oceniany raz, w swoim val-foldzie) -> jedna krzywa/wariant.

Wymaga zapisanych enkoderow (models/representations) i - dla ROC - glowic 'pat' (models/evaluation).
Kontekst (sciezki/urzadzenie) przekazywany argumentami, jak w reszcie src/evaluation/.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.dataloaders import get_fold_dataloaders, get_fold_sequence_dataloaders
from src.evaluation.artifacts import encoder_path, load_head
from src.evaluation.encoders import build_encoder
from src.models.linear_classifier import LinearClassifier


def _load_encoder(variant, model_name, C, T, fold, root, device):
    _state = torch.load(encoder_path(root, variant, fold), map_location=device)
    enc, es = build_encoder(model_name, C, T, device, ckpt_state=_state)  # d_model z wag (Mamba)
    enc.load_state_dict(_state)
    enc.eval()
    return enc, es


# ---------- PCA cech (per okno) ----------
def collect_window_features(variant, model_name, data_dir, root, device, n_splits):
    """Cechy okien ze zbiorow WALIDACYJNYCH wszystkich foldow (enkoder swojego foldu - bez leakage).
    Zwraca (features [N, encoded_size], labels [N]) - po jednym wektorze na okno walidacyjne."""
    feats, labels = [], []
    for fold, _tl, vl in get_fold_dataloaders(data_dir, n_splits=n_splits, batch_size=64):
        s, _ = next(iter(vl)); C, T = s.shape[1], s.shape[2]
        enc, _ = _load_encoder(variant, model_name, C, T, fold, root, device)
        with torch.no_grad():
            for X, y in vl:
                f = enc(X.to(device)); f = f.view(f.size(0), -1)
                feats.append(f.cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels).ravel()


def pca_features_2d(variant, model_name, data_dir, root, device, n_splits, seed=42):
    """Rzut PCA 2D cech okien walidacyjnych. Zwraca (xy [N,2], labels [N], explained_var [2])."""
    X, y = collect_window_features(variant, model_name, data_dir, root, device, n_splits)
    p = PCA(n_components=2, random_state=seed)
    xy = p.fit_transform(X)
    return xy, y, p.explained_variance_ratio_


def plot_pca_grid(variants, data_dir, root, device, n_splits, plt, seed=42):
    """Siatka rzutow PCA dla listy (nazwa, variant, model_name) obok siebie. Kolor = AD(1)/CN(0)."""
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4))
    if len(variants) == 1:
        axes = [axes]
    for ax, (name, variant, model) in zip(axes, variants):
        xy, y, ev = pca_features_2d(variant, model, data_dir, root, device, n_splits, seed=seed)
        for cls, col, lab in [(0, 'tab:blue', 'CN'), (1, 'tab:red', 'AD')]:
            m = (y == cls)
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.35, c=col, label=lab)
        ax.set_title(f'{name}\n(PCA okien val; war. {ev[0]*100:.0f}+{ev[1]*100:.0f}%)', fontsize=9)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8, markerscale=2)
    fig.suptitle('Reprezentacje okien w 2D (PCA) - separowalnosc AD vs CN', fontsize=11)
    fig.tight_layout()
    return fig


# ---------- ROC per-pacjent (pooled po foldach) ----------
def patient_predictions(variant, model_name, data_dir, root, device, n_splits):
    """Predykcje per-pacjent (srednie p-stwo okien) ze wszystkich foldow, POOLED.
    Glowica 'pat' z dysku -> inference. Zwraca (probs [P], labels [P]) dla P=wszyscy pacjenci."""
    probs, labels = [], []
    seq_gen = get_fold_sequence_dataloaders(data_dir, n_splits=n_splits)
    win_gen = get_fold_dataloaders(data_dir, n_splits=n_splits, batch_size=64)
    for (fold, _, vl_seq), (_, tl_win, _) in zip(seq_gen, win_gen):
        s, _ = next(iter(tl_win)); C, T = s.shape[1], s.shape[2]
        enc, es = _load_encoder(variant, model_name, C, T, fold, root, device)
        clf = LinearClassifier(enc, es).to(device)
        clf.classifier.load_state_dict(load_head(root, variant, fold, 'pat', device)); clf.eval()
        with torch.no_grad():
            for seq, label in vl_seq:
                pr = torch.sigmoid(clf(seq.to(device))).cpu().numpy().ravel()
                probs.append(float(pr.mean())); labels.append(int(label))
    return np.array(probs), np.array(labels)


def plot_patient_roc(variants, data_dir, root, device, n_splits, plt):
    """Krzywe ROC per-pacjent (pooled) dla listy (nazwa, variant, model_name) na jednym wykresie."""
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, variant, model in variants:
        probs, labels = patient_predictions(variant, model, data_dir, root, device, n_splits)
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, lw=1.8, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], ls='--', c='k', lw=1, label='losowy (AUC=0.5)')
    ax.set_xlabel('FPR (1 - swoistosc)'); ax.set_ylabel('TPR (czulosc)')
    ax.set_title('Krzywe ROC per-pacjent (pooled po foldach CV)')
    ax.legend(fontsize=8, loc='lower right'); fig.tight_layout()
    return fig
