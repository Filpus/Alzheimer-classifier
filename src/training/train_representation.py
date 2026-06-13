import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import get_fold_dataloaders
from src.models.baseline_ae import Autoencoder

def train_ae():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urzadzenie: {device}")
    
    os.makedirs("models/representations", exist_ok=True)
    
    for fold, train_loader, val_loader in get_fold_dataloaders("data/processed", n_splits=params['n_splits'], batch_size=params['batch_size']):
        print(f"--- FOLD {fold} ---")
        
        sample_batch, _ = next(iter(train_loader))
        num_channels = sample_batch.shape[1]
        sequence_length = sample_batch.shape[2]
        
        model = Autoencoder(num_channels, sequence_length).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0.0
            
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(device)
                
                optimizer.zero_grad()
                _, reconstructed = model(X_batch)
                
                if reconstructed.shape[2] != X_batch.shape[2]:
                    reconstructed = nn.functional.interpolate(reconstructed, size=X_batch.shape[2])
                    
                loss = criterion(reconstructed, X_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    _, reconstructed = model(X_batch)
                    
                    if reconstructed.shape[2] != X_batch.shape[2]:
                        reconstructed = nn.functional.interpolate(reconstructed, size=X_batch.shape[2])
                        
                    loss = criterion(reconstructed, X_batch)
                    val_loss += loss.item()
                    
            print(f"Epoka {epoch+1}/{params['epochs']} | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss/len(val_loader):.4f}")
        
        torch.save(model.encoder.state_dict(), f"models/representations/encoder_fold_{fold}.pth")
        print(f"Zapisano wagi enkodera do models/representations/encoder_fold_{fold}.pth")
        break 

if __name__ == "__main__":
    train_ae()