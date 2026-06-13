import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

class EEGDataset(Dataset):
    def __init__(self, file_list):
        all_tensors = []
        all_labels = []
        
        for f in file_list:
            label = int(os.path.basename(f).split('_label_')[1].replace('.pt', ''))
            tensor = torch.load(f)
            all_tensors.append(tensor)
            all_labels.extend([label] * tensor.shape[0])
            
        self.data = torch.cat(all_tensors, dim=0)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_fold_dataloaders(data_dir, n_splits=5, batch_size=32):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    groups = [os.path.basename(f).split('_')[0] for f in files]
    labels = [int(os.path.basename(f).split('_label_')[1].replace('.pt', '')) for f in files]
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(files, labels, groups=groups)):
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        
        train_dataset = EEGDataset(train_files)
        val_dataset = EEGDataset(val_files)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        yield fold, train_loader, val_loader