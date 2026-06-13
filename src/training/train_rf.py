import os
import yaml
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import get_fold_dataloaders
from src.models.baseline_ae import Autoencoder

def train_rf():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, train_loader, val_loader in get_fold_dataloaders("data/processed", n_splits=params['n_splits'], batch_size=params['batch_size']):
        print(f"\n--- TRENOWANIE KLASYFIKATORA RF: FOLD {fold} ---")

        sample_batch, _ = next(iter(train_loader))
        num_channels = sample_batch.shape[1]
        sequence_length = sample_batch.shape[2]

        base_ae = Autoencoder(num_channels, sequence_length).to(device)
        encoder_path = f"models/representations/encoder_fold_{fold}.pth"
        base_ae.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        base_ae.eval()

        X_train_features = []
        y_train_labels = []
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                features = base_ae.encoder(X_batch)
                features = features.view(features.size(0), -1).cpu().numpy()
                X_train_features.extend(features)
                y_train_labels.extend(y_batch.numpy())

        X_val_features = []
        y_val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                features = base_ae.encoder(X_batch)
                features = features.view(features.size(0), -1).cpu().numpy()
                X_val_features.extend(features)
                y_val_labels.extend(y_batch.numpy())

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train_features, y_train_labels)

        preds = clf.predict(X_val_features)
        probs = clf.predict_proba(X_val_features)[:, 1]

        val_acc = accuracy_score(y_val_labels, preds)
        val_auc = roc_auc_score(y_val_labels, probs)

        print(f"Random Forest | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        break

if __name__ == "__main__":
    train_rf()