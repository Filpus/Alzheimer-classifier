"""Ewaluacja PER-PACJENT: agregacja predykcji okien pacjenta do jednej decyzji.

Diagnoza dotyczy pacjenta, a okna jednego pacjenta nie sa niezalezne - usredniamy
prawdopodobienstwa okien danego pacjenta (-> 1 predykcja/pacjent) i liczymy metryki na
poziomie pacjentow. Metryki z najlepszej epoki wg AUC per-pacjent + early-stopping
(spojnie z definicja 'best po epokach' dla okna w linear_evaluation).

Trzy sciezki (wg load_probe): (1) kompletny cache JSON -> wczytaj; (2) zapisana GLOWICA 'pat'
-> inference bez treningu; (3) trening glowicy + zapis 'pat'. Per-okno NIE liczymy tutaj -
do tabeli per-okno bierze sie gotowe wyniki sekcji 4/5 (PERWIN_FILE).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data.dataloaders import get_fold_dataloaders, get_fold_sequence_dataloaders
from src.evaluation.artifacts import all_heads_exist, encoder_path, load_head, save_head
from src.evaluation.encoders import build_encoder
from src.evaluation.results_cache import load_json, results_exist, save_json
from src.models.linear_classifier import LinearClassifier

# wariant -> plik wynikow per-OKNO z sekcji 4/5 (te same liczby co linear-eval tam).
PERWIN_FILE = {'ae': 'ae_default.json',
               'tnc': 'tnc_default.json', 'tnc_best': 'tnc_best.json',
               'cpc': 'cpc_default.json', 'cpc_best': 'cpc_best.json'}
# komplet kluczy cache per-PACJENT (do walidacji starego/niekompletnego cache).
PP_KEYS = [f'pat_{m}' for m in ('auc', 'acc', 'f1', 'prec', 'rec')]


def patient_metrics(clf, vl_seq, device):
    """Inference per-pacjent: srednie prawdopodobienstwo okien pacjenta -> 1 predykcja/pacjent.
    Zwraca dict metryk (AUC/Acc/F1/Prec/Recall, klasa AD=1). clf w trybie eval przed wywolaniem."""
    pp, pl = [], []
    with torch.no_grad():
        for seq, label in vl_seq:
            pr = torch.sigmoid(clf(seq.to(device))).cpu().numpy().ravel()
            pp.append(float(pr.mean())); pl.append(int(label))
    pred = [int(x > 0.5) for x in pp]
    return {'auc': roc_auc_score(pl, pp), 'acc': accuracy_score(pl, pred),
            'f1': f1_score(pl, pred, zero_division=0), 'prec': precision_score(pl, pred, zero_division=0),
            'rec': recall_score(pl, pred, zero_division=0)}


def eval_per_patient(variant, model_name, load_probe, data_dir, root, results_dir, device,
                     n_splits, probe_epochs, probe_patience, res_file=None):
    """Metryki per-pacjent dla zapisanego enkodera wariantu (patrz docstring modulu).
    Zwraca dict list per-fold: pat_auc/pat_acc/pat_f1/pat_prec/pat_rec."""
    # (1) gotowy, kompletny cache liczb
    if load_probe and res_file and results_exist(results_dir, res_file):
        cached = load_json(results_dir, res_file)
        if all(k in cached for k in PP_KEYS):
            print(f'[{variant}] Wczytano zapisany per-patient ({res_file}).')
            return cached
        print(f'[{variant}] Stary/niekompletny cache ({res_file}) -> przeliczam.')
    use_heads = load_probe and all_heads_exist(root, variant, n_splits, 'pat')  # (2) inference z glowic
    pat = {'auc': [], 'acc': [], 'f1': [], 'prec': [], 'rec': []}
    seq_gen = get_fold_sequence_dataloaders(data_dir, n_splits=n_splits)
    win_gen = get_fold_dataloaders(data_dir, n_splits=n_splits, batch_size=64)
    for (fold, _, vl_seq), (_, tl_win, _) in zip(seq_gen, win_gen):
        s, _ = next(iter(tl_win)); C, T = s.shape[1], s.shape[2]
        enc, es = build_encoder(model_name, C, T, device)
        enc.load_state_dict(torch.load(encoder_path(root, variant, fold), map_location=device)); enc.eval()
        clf = LinearClassifier(enc, es).to(device)   # enkoder zamrozony w LinearClassifier
        if use_heads:
            # (2) wczytaj glowice 'pat' i policz metryki inference (bez treningu)
            clf.classifier.load_state_dict(load_head(root, variant, fold, 'pat', device)); clf.eval()
            best_p = patient_metrics(clf, vl_seq, device)
            print(f'[{variant}] FOLD {fold}: glowica pat z dysku -> inference (AUC {best_p["auc"]:.4f})')
        else:
            # (3) trening glowicy z early-stoppingiem na pat_auc; zapamietaj best-epoka state_dict
            crit = nn.BCEWithLogitsLoss(); opt = optim.Adam(clf.classifier.parameters(), lr=5e-3)
            best_p = {'auc': -1.0}; best_head = copy.deepcopy(clf.classifier.state_dict()); no_improve = 0
            for _ in range(probe_epochs):
                clf.train()
                for X, y in tl_win:
                    X = X.to(device); y = y.float().to(device).unsqueeze(1)
                    opt.zero_grad(); crit(clf(X), y).backward(); opt.step()
                clf.eval()
                p = patient_metrics(clf, vl_seq, device)
                if p['auc'] > best_p['auc']:
                    best_p = p; best_head = copy.deepcopy(clf.classifier.state_dict()); no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= probe_patience:   # brak poprawy pat_auc przez patience epok -> stop
                        break
            save_head(best_head, root, variant, fold, 'pat')
            print(f'[{variant}] FOLD {fold}: trening+zapis glowicy pat (AUC {best_p["auc"]:.4f})')
        for k in pat:
            pat[k].append(best_p[k])
    out = {f'pat_{k}': v for k, v in pat.items()}
    if res_file:
        save_json(out, results_dir, res_file)
    return out
