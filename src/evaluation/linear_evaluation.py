"""Protokol linear evaluation na walidacji krzyzowej (per-okno).

Rdzen oceny reprezentacji: zamrozony enkoder + plytka glowica liniowa, AUC/Acc per-okno
usredniane po foldach StratifiedGroupKFold (na poziomie pacjenta). Funkcje realizuja
schemat TRAIN-OR-LOAD sterowany dwiema flagami:

  load_probe   - True i sa zapisane wyniki linear-eval -> wczytaj liczby (bez probe i SSL);
                 w przeciwnym razie probe: sa zapisane GLOWICE 'win' -> inference, inaczej trening.
  load_encoder - jesli probe trzeba przeliczyc: True i sa wagi -> wczytaj enkoder (bez SSL),
                 inaczej trenuj SSL od zera i zapisz wagi.

Glowice (epoka max AUC per-okno) zapisywane sa do models/evaluation/<wariant>/head_win_*.pth,
co pozwala kolejnym przebiegom liczyc metryki samym inference. Early-stopping (probe_patience)
dziala na KAZDEJ ewaluacji liniowej. Wszystkie sciezki/urzadzenie przekazywane sa argumentami.

Rdzen treningu (train_ssl, linear_probe, eval_head) jest w src/training/ssl_training.py -
to samo zrodlo, ktorego uzywaja skrypty CLI; tu tylko warstwa CV + cache + warianty 'best'.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloaders import get_fold_dataloaders, get_fold_sequence_dataloaders
from src.evaluation.artifacts import all_encoders_exist, all_heads_exist, encoder_path, load_head, save_encoder, save_head
from src.evaluation.encoders import build_encoder
from src.evaluation.results_cache import load_json, results_exist, save_json
from src.models.baseline_ae import Autoencoder
from src.training.ssl_training import eval_head, linear_probe, train_ssl


# ---------- pelna CV z treningiem SSL ----------
def run_cv(model_factory, data_dir, root, device, ssl_epochs, n_splits, probe_epochs,
           probe_patience, lr_ssl=1e-3, verbose=True, save_variant=None):
    """Dla kazdego foldu: trening SSL na sekwencjach, klasyfikacja liniowa na oknach.
    Gdy save_variant podany - zapisuje enkoder oraz glowice 'win' (epoka max AUC) do models/.
    Zwraca (accs, aucs, hists)."""
    accs, aucs, hists = [], [], []
    seq_gen = get_fold_sequence_dataloaders(data_dir, n_splits=n_splits)
    win_gen = get_fold_dataloaders(data_dir, n_splits=n_splits, batch_size=64)
    for (fold, tl_seq, _), (_, tl_win, vl_win) in zip(seq_gen, win_gen):
        s, _ = next(iter(tl_seq)); C, T = s.shape[1], s.shape[2]
        if verbose:
            print(f'--- FOLD {fold} (trening) ---')
        m = model_factory(C, T)
        h = train_ssl(m, tl_seq, ssl_epochs, device, lr=lr_ssl, verbose=verbose)
        if save_variant:
            save_encoder(m, root, save_variant, fold)
        m.encoder.eval()
        acc, auc, head = linear_probe(m.encoder, m.encoded_size, tl_win, vl_win, probe_epochs,
                                      device, patience=probe_patience)
        if save_variant:
            save_head(head, root, save_variant, fold, 'win')
        if verbose:
            print(f'  -> linear-eval: Acc {acc:.4f} | AUC {auc:.4f}')
        accs.append(acc); aucs.append(auc); hists.append(h)
    return accs, aucs, hists


# ---------- CV na gotowych enkoderach (bez SSL) ----------
def run_cv_pretrained(variant, model_name, data_dir, root, device, n_splits, probe_epochs,
                      probe_patience, load_probe=True, verbose=True):
    """Wczytuje gotowe enkodery. Jesli load_probe i sa zapisane GLOWICE 'win' -> inference
    bez treningu; inaczej dotrenuj glowice i zapisz. Zwraca (accs, aucs, [])."""
    use_heads = load_probe and all_heads_exist(root, variant, n_splits, 'win')
    accs, aucs = [], []
    for fold, tl_win, vl_win in get_fold_dataloaders(data_dir, n_splits=n_splits, batch_size=64):
        s, _ = next(iter(tl_win)); C, T = s.shape[1], s.shape[2]
        _state = torch.load(encoder_path(root, variant, fold), map_location=device)
        enc, es = build_encoder(model_name, C, T, device, ckpt_state=_state)  # d_model z wag (Mamba)
        enc.load_state_dict(_state); enc.eval()
        if use_heads:
            acc, auc = eval_head(enc, es, load_head(root, variant, fold, 'win', device), vl_win, device)
            if verbose:
                print(f'--- FOLD {fold} (glowica win z dysku -> inference) Acc {acc:.4f} | AUC {auc:.4f}')
        else:
            acc, auc, head = linear_probe(enc, es, tl_win, vl_win, probe_epochs, device, patience=probe_patience)
            save_head(head, root, variant, fold, 'win')
            if verbose:
                print(f'--- FOLD {fold} (enkoder {variant}, probe+zapis glowicy) Acc {acc:.4f} | AUC {auc:.4f}')
        accs.append(acc); aucs.append(auc)
    return accs, aucs, []


# ---------- train-or-load ----------
def run_cv_smart(model_factory, model_name, variant, data_dir, root, results_dir, device,
                 ssl_epochs, n_splits, probe_epochs, probe_patience,
                 load_encoder, load_probe, lr_ssl=1e-3, hist_file=None, res_file=None):
    """Train-or-load (patrz docstring modulu). Po przeliczeniu probe zapisuje wyniki (i historie,
    gdy byl trening SSL); glowice zapisuja run_cv*. Zwraca (accs, aucs, hist)."""
    # (1) gotowe wyniki linear-eval -> najtansza sciezka
    if load_probe and res_file and results_exist(results_dir, res_file):
        r = load_json(results_dir, res_file)
        hist = load_json(results_dir, hist_file) if (hist_file and results_exist(results_dir, hist_file)) else []
        print(f'[{variant.upper()}] Wczytano zapisane wyniki linear-eval ({res_file}) - bez probe i treningu.')
        return r['accs'], r['aucs'], hist
    # (2) probe; enkoder z wag albo trening SSL
    if load_encoder and all_encoders_exist(root, variant, n_splits):
        print(f'[{variant.upper()}] Wczytuje gotowe enkodery (wagi) -> klasyfikacja liniowa (glowice z dysku lub trening).')
        accs, aucs, hist = run_cv_pretrained(variant, model_name, data_dir, root, device, n_splits,
                                             probe_epochs, probe_patience, load_probe=load_probe)
    else:
        if load_encoder:
            print(f'[{variant.upper()}] Brak kompletu wag na dysku -> trening SSL od zera.')
        else:
            print(f'[{variant.upper()}] LOAD_ENCODER=False -> trening SSL od zera.')
        accs, aucs, hist = run_cv(model_factory, data_dir, root, device, ssl_epochs, n_splits,
                                  probe_epochs, probe_patience, lr_ssl=lr_ssl, save_variant=variant)
    if res_file:
        save_json({'accs': accs, 'aucs': aucs}, results_dir, res_file)
    if hist_file and hist:
        save_json(hist, results_dir, hist_file)  # historia tylko gdy byl trening SSL
    return accs, aucs, hist


# ---------- ocena konfiguracji HP (bez zapisu wag/glowic) ----------
def eval_n_folds(model_factory, data_dir, root, device, ssl_epochs, probe_epochs, n_folds, probe_patience, lr_ssl=1e-3):
    """Srednie (acc, auc, std_auc) po n_folds dla danej konfiguracji - bez zapisu artefaktow."""
    import numpy as np
    accs, aucs, _ = run_cv(model_factory, data_dir, root, device, ssl_epochs, n_folds, probe_epochs,
                           probe_patience, lr_ssl=lr_ssl, verbose=False, save_variant=None)
    return float(np.mean(accs)), float(np.mean(aucs)), float(np.std(aucs))


# ---------- baseline AE (trening rekonstrukcyjny, osobno od SSL) ----------
def train_cv_ae(data_dir, root, device, epochs, n_splits, probe_epochs, probe_patience, lr=1e-3):
    """Trenuje Autoenkoder rekonstrukcja (MSE), zapisuje enkoder + glowice 'win', robi linear-eval.
    Osobno od run_cv, bo AE uczy sie rekonstrukcja, nie zadaniem SSL. Zwraca (accs, aucs)."""
    accs, aucs = [], []
    for fold, tl, vl in get_fold_dataloaders(data_dir, n_splits=n_splits, batch_size=64):
        s, _ = next(iter(tl)); C, T = s.shape[1], s.shape[2]
        print(f'--- FOLD {fold} (AE - trening) ---')
        ae = Autoencoder(C, T).to(device)
        crit = nn.MSELoss(); opt = optim.Adam(ae.parameters(), lr=lr)
        for _ in range(epochs):
            ae.train()
            for X, _ in tl:
                X = X.to(device); opt.zero_grad(); _, rec = ae(X)
                if rec.shape[2] != X.shape[2]:
                    rec = nn.functional.interpolate(rec, size=X.shape[2])
                crit(rec, X).backward(); opt.step()
        save_encoder(ae, root, 'ae', fold)
        ae.encoder.eval()
        acc, auc, head = linear_probe(ae.encoder, ae.encoded_size, tl, vl, probe_epochs, device, patience=probe_patience)
        save_head(head, root, 'ae', fold, 'win')
        print(f'  -> linear-eval: Acc {acc:.4f} | AUC {auc:.4f}')
        accs.append(acc); aucs.append(auc)
    return accs, aucs


def run_cv_ae(data_dir, root, results_dir, device, epochs, n_splits, probe_epochs, probe_patience,
              load_encoder, load_probe, lr=1e-3, res_file='ae_default.json'):
    """Train-or-load AE - te same dwie flagi co run_cv_smart. Zwraca (accs, aucs)."""
    if load_probe and results_exist(results_dir, res_file):
        r = load_json(results_dir, res_file)
        print(f'[AE] Wczytano zapisane wyniki linear-eval ({res_file}) - bez probe i treningu.')
        return r['accs'], r['aucs']
    if load_encoder and all_encoders_exist(root, 'ae', n_splits):
        print('[AE] Wczytuje gotowe enkodery (wagi) -> klasyfikacja liniowa (glowice z dysku lub trening).')
        accs, aucs, _ = run_cv_pretrained('ae', 'ae', data_dir, root, device, n_splits, probe_epochs,
                                          probe_patience, load_probe=load_probe)
    else:
        if load_encoder:
            print('[AE] Brak kompletu wag na dysku -> trening AE od zera.')
        else:
            print('[AE] AE_LOAD_ENCODER=False -> trening AE od zera.')
        accs, aucs = train_cv_ae(data_dir, root, device, epochs, n_splits, probe_epochs, probe_patience, lr=lr)
    save_json({'accs': accs, 'aucs': aucs}, results_dir, res_file)
    return accs, aucs
