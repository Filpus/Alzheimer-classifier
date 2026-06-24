import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold


def _label_from_filename(path):
    return int(os.path.basename(path).split('_label_')[1].replace('.pt', ''))


def _subject_from_filename(path):
    return os.path.basename(path).split('_')[0]


class EEGDataset(Dataset):
    """Plaska kolekcja pojedynczych okien EEG (window-level).

    Skleja okna wszystkich pacjentow w jeden tensor. Uzywany przez baseline
    (autoenkoder) oraz protokol linear evaluation, gdzie kolejnosc czasowa
    okien nie ma znaczenia.
    """

    def __init__(self, file_list):
        all_tensors = []
        all_labels = []

        for f in file_list:
            label = _label_from_filename(f)
            tensor = torch.load(f)
            all_tensors.append(tensor)
            all_labels.extend([label] * tensor.shape[0])

        self.data = torch.cat(all_tensors, dim=0)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class EEGSequenceDataset(Dataset):
    """Kolekcja na poziomie pacjenta, zachowujaca kolejnosc czasowa okien.

    Kazdy element to caly zapis jednego pacjenta o ksztalcie
    (num_windows, num_channels, window_samples). Okna NIE sa tasowane wewnatrz
    pacjenta, dzieki czemu mozliwe jest uczenie samonadzorowane wykorzystujace
    sasiedztwo czasowe: CPC (predykcja przyszlych okien) oraz TNC (sasiedztwo
    czasowe vs okna odlegle).

    Ze wzgledu na rozna liczbe okien u kazdego pacjenta uzywaj batch_size=1
    (lub wlasnego collate_fn). Liczba okien w jednym pacjencie (47-225)
    wystarcza jako "batch" dla strat kontrastywnych.
    """

    def __init__(self, file_list):
        self.file_list = list(file_list)
        self.labels = [_label_from_filename(f) for f in self.file_list]
        self.subjects = [_subject_from_filename(f) for f in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # (num_windows, num_channels, window_samples) w oryginalnej kolejnosci czasowej
        seq = torch.load(self.file_list[idx])
        return seq, self.labels[idx]


def _fold_splits(data_dir, n_splits):
    """Wspolny podzial StratifiedGroupKFold na poziomie pacjenta.

    Zwraca te same indeksy foldow dla wszystkich modeli, dzieki czemu
    porownanie AE / CPC / TNC jest przeprowadzane na identycznym podziale.
    """
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
    files = sorted(files)  # deterministyczna kolejnosc plikow

    groups = [_subject_from_filename(f) for f in files]
    labels = [_label_from_filename(f) for f in files]

    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(files, labels, groups=groups)):
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        yield fold, train_files, val_files


def get_fold_dataloaders(data_dir, n_splits=5, batch_size=32):
    """Window-level dataloadery (baseline + linear evaluation)."""
    for fold, train_files, val_files in _fold_splits(data_dir, n_splits):
        train_dataset = EEGDataset(train_files)
        val_dataset = EEGDataset(val_files)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        yield fold, train_loader, val_loader


def _sequence_collate(batch):
    # batch_size=1: rozpakowuje pojedynczego pacjenta bez dodatkowego wymiaru batcha
    seq, label = batch[0]
    return seq, label


def get_fold_sequence_dataloaders(data_dir, n_splits=5):
    """Patient-level dataloadery zachowujace kolejnosc czasowa (CPC / TNC).

    Iteracja po loaderze zwraca (seq, label), gdzie seq ma ksztalt
    (num_windows, num_channels, window_samples) dla jednego pacjenta.
    """
    for fold, train_files, val_files in _fold_splits(data_dir, n_splits):
        train_dataset = EEGSequenceDataset(train_files)
        val_dataset = EEGSequenceDataset(val_files)

        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, collate_fn=_sequence_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=_sequence_collate
        )

        yield fold, train_loader, val_loader