import os
import yaml
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import get_fold_sequence_dataloaders
from src.models.tnc import TNCModel
from src.evaluation.encoders import encoder_dir, encoder_path
from src.training.ssl_training import train_ssl


def train_tnc():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config['train']
    tnc_params = config.get('tnc', {})

    epochs = tnc_params.get('epochs', params['epochs'])
    lr = tnc_params.get('learning_rate', params['learning_rate'])
    neighbor_range = tnc_params.get('neighbor_range', 3)
    num_samples = tnc_params.get('num_samples', 5)
    w = tnc_params.get('w', 0.05)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urzadzenie: {device}")

    os.makedirs(encoder_dir("tnc"), exist_ok=True)

    for fold, train_loader, _ in get_fold_sequence_dataloaders(
        "data/processed", n_splits=params['n_splits']
    ):
        print(f"\n--- TRENOWANIE REPREZENTACJI TNC: FOLD {fold} ---")

        sample_seq, _ = next(iter(train_loader))
        num_channels = sample_seq.shape[1]
        sequence_length = sample_seq.shape[2]

        model = TNCModel(
            num_channels, sequence_length,
            neighbor_range=neighbor_range, num_samples=num_samples, w=w,
        )
        train_ssl(model, train_loader, epochs, device, lr=lr, loss_name="TNC")

        out_path = encoder_path("tnc", fold)
        torch.save(model.encoder.state_dict(), out_path)
        print(f"Zapisano wagi enkodera TNC do {out_path}")


if __name__ == "__main__":
    train_tnc()
