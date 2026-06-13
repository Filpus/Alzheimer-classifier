import os
import yaml
from abc import ABC, abstractmethod
import mne
import pandas as pd
import numpy as np
import torch

class BaseEEGProcessor(ABC):
    def __init__(self, raw_dir, processed_dir, params):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.params = params

    @abstractmethod
    def load_metadata(self):
        pass

    @abstractmethod
    def process_and_save(self, metadata):
        pass

class BrainLatProcessor(BaseEEGProcessor):
    def load_metadata(self):
        ad_path = os.path.join(self.raw_dir, 'metadata', 'Records_AD_EEG_data.csv')
        hc_path = os.path.join(self.raw_dir, 'metadata', 'Records_HC_EEG_data.csv')
        
        ad_df = pd.read_csv(ad_path)
        hc_df = pd.read_csv(hc_path)
        
        ad_df = ad_df[['id_EEG', 'diagnosis']].copy()
        hc_df = hc_df[['id_EEG', 'diagnosis']].copy()
        
        ad_df['label'] = 1
        hc_df['label'] = 0
        
        return pd.concat([ad_df, hc_df], ignore_index=True)

    def process_and_save(self, metadata):
        os.makedirs(self.processed_dir, exist_ok=True)
        eeg_dir = os.path.join(self.raw_dir, 'data')
        
        if not os.path.exists(eeg_dir):
            print(f"BŁĄD KRYTYCZNY: Nie znaleziono folderu z plikami EEG: {eeg_dir}")
            return
            
        all_files = []
        for root, dirs, files in os.walk(eeg_dir):
            for file in files:
                if file.endswith('.set'):
                    all_files.append(os.path.join(root, file))
                    
        print(f"Znaleziono {len(all_files)} plików .set w całym drzewie folderów.")
        
        for _, row in metadata.iterrows():
            subject_id = str(row['id_EEG']).strip()
            label = row['label']
            diagnosis = str(row['diagnosis']).strip()
            
            matching_files = [f for f in all_files if f"{subject_id}_" in os.path.basename(f)]
            
            if not matching_files:
                print(f"POMINIĘTO [{subject_id}]: Brak pasującego pliku .set")
                continue
                
            file_path = matching_files[0]
            
            try:
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            except Exception as e:
                print(f"BŁĄD MNE [{subject_id}]: {e}")
                continue
                
            sfreq = raw.info['sfreq']
            window_samples = int(self.params['window_size'] * sfreq)
            
            data = raw.get_data()
            
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            data_normalized = data 
            
            num_windows = data_normalized.shape[1] // window_samples
            if num_windows == 0:
                print(f"BŁĄD DŁUGOŚCI [{subject_id}]: Sygnał zbyt krótki")
                continue
                
            data_trimmed = data_normalized[:, :num_windows * window_samples]
            epochs = data_trimmed.reshape(data_normalized.shape[0], num_windows, window_samples)
            epochs = np.transpose(epochs, (1, 0, 2))
            
            tensor_data = torch.tensor(epochs, dtype=torch.float32)
            output_file = os.path.join(self.processed_dir, f"{subject_id}_{diagnosis}_label_{label}.pt")
            
            torch.save(tensor_data, output_file)
            print(f"SUKCES [{subject_id}]: Zapisano {num_windows} epok ({diagnosis}).")

class DatasetFactory:
    @staticmethod
    def get_processor(dataset_name, raw_dir, processed_dir, params):
        if dataset_name == 'brainlat':
            return BrainLatProcessor(raw_dir, processed_dir, params)
        raise ValueError(f"Dataset {dataset_name} is not supported.")

def main():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    params = config.get('make_dataset', {})
    
    processor = DatasetFactory.get_processor(
        dataset_name=params.get('dataset_name', 'brainlat'),
        raw_dir=params.get('raw_dir', 'data/raw/brainlat'),
        processed_dir=params.get('processed_dir', 'data/processed'),
        params=params
    )
    
    metadata = processor.load_metadata()
    processor.process_and_save(metadata)

if __name__ == "__main__":
    main()