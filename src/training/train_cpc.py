import os
import yaml
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import get_fold_sequence_dataloaders
from src.models.cpc import CPCModel
from src.evaluation.encoders import encoder_dir, encoder_path
from src.training.ssl_training import train_ssl


def train_cpc():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config['train']
    cpc_params = config.get('cpc', {})

    epochs = cpc_params.get('epochs', params['epochs'])
    lr = cpc_params.get('learning_rate', params['learning_rate'])
    context_dim = cpc_params.get('context_dim', 128)
    prediction_steps = cpc_params.get('prediction_steps', 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urzadzenie: {device}")

    os.makedirs(encoder_dir("cpc"), exist_ok=True)

    for fold, train_loader, _ in get_fold_sequence_dataloaders(
        "data/processed", n_splits=params['n_splits']
    ):
        print(f"\n--- TRENOWANIE REPREZENTACJI CPC: FOLD {fold} ---")

        sample_seq, _ = next(iter(train_loader))
        num_channels = sample_seq.shape[1]
        sequence_length = sample_seq.shape[2]

        model = CPCModel(
            num_channels, sequence_length,
            context_dim=context_dim, prediction_steps=prediction_steps,
        )
        train_ssl(model, train_loader, epochs, device, lr=lr, loss_name="InfoNCE")

        out_path = encoder_path("cpc", fold)
        torch.save(model.encoder.state_dict(), out_path)
        print(f"Zapisano wagi enkodera CPC do {out_path}")


if __name__ == "__main__":
    train_cpc()
