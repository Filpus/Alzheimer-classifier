"""Dobor hiperparametrow reprezentacji - dwuetapowy, o roznym koszcie.

  hp_study   - badanie WPLYWU jednego HP na hp_splits foldach (tanio): pokazuje trend
               i wylania 2 najlepsze wartosci (top2) do siatki.
  grid_search - mala siatka kombinacji (2x2) na PELNYM n_splits CV - finalny, deterministyczny
               wybor zastepujacy random search.

Oba etapy TRENUJA enkodery od zera (przez eval_n_folds), wiec wczytywaniem steruje flaga
enkodera (load_encoder): True i tabela istnieje -> wczytaj zapisana; inaczej policz i zapisz.
Kontekst (sciezki/urzadzenie/epoki) przekazywany jest argumentami.
"""

import pandas as pd

from src.evaluation.linear_evaluation import eval_n_folds
from src.evaluation.results_cache import load_df, results_exist, save_df

METRIC_COLS = ('Accuracy', 'ROC-AUC', 'ROC-AUC_std')


def hp_study(build_model, param_name, grid, fixed, res_file, load_encoder,
             data_dir, root, results_dir, device, ssl_epochs, probe_epochs, hp_splits, probe_patience):
    """Badanie wplywu `param_name` (wartosci `grid`, pozostale HP w `fixed`) na hp_splits foldach.
    Zwraca DataFrame (wartosc HP + Accuracy/ROC-AUC/ROC-AUC_std), cache w res_file."""
    if load_encoder and results_exist(results_dir, res_file):
        print(f'Wczytano zapisana tabele badania HP ({res_file}).')
        return load_df(results_dir, res_file)
    rows = []
    for val in grid:
        print(f'[{param_name} = {val}] ({hp_splits}-fold CV...)')
        params = {**fixed, param_name: val}
        acc, auc, sd = eval_n_folds(lambda C, T, p=params: build_model(C, T, **p),
                                    data_dir, root, device, ssl_epochs, probe_epochs, hp_splits, probe_patience)
        rows.append({param_name: val, 'Accuracy': round(acc, 4), 'ROC-AUC': round(auc, 4), 'ROC-AUC_std': round(sd, 4)})
        print(f'    -> AUC {auc:.4f} +/- {sd:.4f}')
    df = pd.DataFrame(rows)
    save_df(df, results_dir, res_file)
    return df


def top2(hp_df, param_name):
    """Dwie wartosci `param_name` o najwyzszym ROC-AUC z tabeli badania HP (do siatki 2x2)."""
    return hp_df.sort_values('ROC-AUC', ascending=False)[param_name].tolist()[:2]


def grid_search(build_model, grid_dict, res_file, load_encoder,
                data_dir, root, results_dir, device, ssl_epochs, probe_epochs, n_splits, probe_patience):
    """Pelna siatka kombinacji (tu 2x2) na pelnym n_splits CV. Zwraca DataFrame posortowany
    malejaco po ROC-AUC, cache w res_file."""
    if load_encoder and results_exist(results_dir, res_file):
        print(f'Wczytano zapisana tabele siatki HP ({res_file}).')
        return load_df(results_dir, res_file)
    import itertools
    keys = list(grid_dict.keys()); rows = []
    for combo in itertools.product(*[grid_dict[k] for k in keys]):
        params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in zip(keys, combo)}
        acc, auc, sd = eval_n_folds(lambda C, T, p=params: build_model(C, T, **p),
                                    data_dir, root, device, ssl_epochs, probe_epochs, n_splits, probe_patience)
        rows.append({**params, 'Accuracy': round(acc, 4), 'ROC-AUC': round(auc, 4), 'ROC-AUC_std': round(sd, 4)})
        print(f'  {params} -> AUC {auc:.4f} +/- {sd:.4f}')
    df = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    save_df(df, results_dir, res_file)
    return df


def best_params_from(rs_df, int_keys):
    """Najlepszy zestaw HP z tabeli siatki (wiersz 0, bez kolumn metryk), rzutujac int_keys na int."""
    p = {k: rs_df.iloc[0][k] for k in rs_df.columns if k not in METRIC_COLS}
    for k in int_keys:
        if k in p:
            p[k] = int(p[k])
    return p
