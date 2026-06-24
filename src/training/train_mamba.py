import os
import yaml
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import get_fold_sequence_dataloaders
from src.models.mamba_model import MambaModel
from src.evaluation.encoders import encoder_dir, encoder_path
from src.training.ssl_training import train_ssl


def _select_device():
    """Wybiera najlepsze dostepne urzadzenie: CUDA > MPS (Apple Silicon) > CPU.

    Selektywny SSM w mamba_model.py jest implementacja w czystym PyTorch (bez
    customowych kernelow CUDA/Triton), dziala wiec identycznie na kazdym z nich.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_mamba():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    params = config['train']
    mamba_params = config.get('mamba', {})

    epochs = mamba_params.get('epochs', 5)
    lr = mamba_params.get('learning_rate', 0.001)
    d_model = mamba_params.get('d_model', 64)
    d_state = mamba_params.get('d_state', 16)

    device = _select_device()
    print(f"Urzadzenie: {device}")

    os.makedirs(encoder_dir("mamba"), exist_ok=True)

    for fold, train_loader, _ in get_fold_sequence_dataloaders(
        "data/processed", n_splits=params['n_splits']
    ):
        print(f"\n--- TRENOWANIE REPREZENTACJI MAMBA: FOLD {fold} ---")

        sample_seq, _ = next(iter(train_loader))
        num_channels = sample_seq.shape[1]
        sequence_length = sample_seq.shape[2]

        model = MambaModel(
            num_channels, sequence_length,
            d_model=d_model, d_state=d_state,
        )
        train_ssl(model, train_loader, epochs, device, lr=lr, loss_name="InfoNCE")

        out_path = encoder_path("mamba", fold)
        torch.save(model.encoder.state_dict(), out_path)
        print(f"Zapisano wagi enkodera Mamba do {out_path}")


if __name__ == "__main__":
    train_mamba()