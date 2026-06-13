import torch
import numpy as np
from src.data.dataloaders import get_fold_dataloaders
from src.models.baseline_ae import Autoencoder

def check_feature_dist():
    # Pobierz przykładowe dane
    for _, train_loader, _ in get_fold_dataloaders("data/processed", n_splits=5, batch_size=128):
        batch, labels = next(iter(train_loader))
        
        # Oblicz średnią wartość całego sygnału dla każdej klasy
        class_0 = batch[labels == 0].mean()
        class_1 = batch[labels == 1].mean()
        
        print(f"Średnia sygnału dla klasy 0: {class_0:.4f}")
        print(f"Średnia sygnału dla klasy 1: {class_1:.4f}")
        break

if __name__ == "__main__":
    check_feature_dist()