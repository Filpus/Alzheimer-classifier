"""Lekki cache wynikow liczbowych ewaluacji (-> git).

Historie strat, tabele HP/siatki, metryki CV i per-pacjent zapisywane sa jako
JSON/CSV w katalogu wynikow (notebooks/results). To wystarcza, by odtworzyc raport
bez kosztownego treningu, gdy ktos nie pobral wag z DVC.

Wszystkie funkcje przyjmuja `results_dir` jako argument - brak stanu globalnego.
"""

import json
import os

import pandas as pd


def res_path(results_dir, name):
    return os.path.join(results_dir, name)


def save_json(obj, results_dir, name):
    with open(res_path(results_dir, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def load_json(results_dir, name):
    with open(res_path(results_dir, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def save_df(df, results_dir, name):
    df.to_csv(res_path(results_dir, name), index=False)


def load_df(results_dir, name):
    return pd.read_csv(res_path(results_dir, name))


def results_exist(results_dir, *names):
    return all(os.path.exists(res_path(results_dir, n)) for n in names)
