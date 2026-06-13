import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.dataloaders import get_fold_dataloaders
from src.models.baseline_ae import Autoencoder
from src.models.linear_classifier import LinearClassifier

def train_classifier():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, train_loader, val_loader in get_fold_dataloaders("data/processed", n_splits=params['n_splits'], batch_size=params['batch_size']):
        print(f"\n--- TRENOWANIE KLASYFIKATORA: FOLD {fold} ---")
        
        sample_batch, _ = next(iter(train_loader))
        num_channels = sample_batch.shape[1]
        sequence_length = sample_batch.shape[2]
        
        base_ae = Autoencoder(num_channels, sequence_length)
        encoder_path = f"models/representations/encoder_fold_{fold}.pth"
        base_ae.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        
        model = LinearClassifier(base_ae.encoder, base_ae.encoded_size).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.005) 
        
        for epoch in range(10):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            all_probs = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.float().to(device).unsqueeze(1)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())
                    
            val_acc = accuracy_score(all_targets, all_preds)
            val_auc = roc_auc_score(all_targets, all_probs)
            
            print(f"Epoka {epoch+1}/10 | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
            
        break

if __name__ == "__main__":
    train_classifier()