"""Prezentacja wynikow ewaluacji: formatowanie metryk i wykres historii treningu."""

import matplotlib.pyplot as plt
import numpy as np


def fmt(vals):
    """'srednia +/- std' po foldach, do tabel raportu."""
    return f'{np.mean(vals):.3f} ± {np.std(vals):.3f}'


def plot_hist(hists, title, msg):
    """Wykres krzywych straty SSL per fold. Gdy brak historii (wczytano gotowe wyniki)
    wypisuje `msg` zamiast pustego wykresu."""
    if hists:
        plt.figure(figsize=(6, 3.3))
        for i, h in enumerate(hists):
            plt.plot(range(1, len(h) + 1), h, marker='o', alpha=0.7, label=f'fold {i}')
        plt.title(title); plt.xlabel('epoka'); plt.ylabel('loss')
        plt.legend(fontsize=8); plt.tight_layout(); plt.show()
    else:
        print(msg)
