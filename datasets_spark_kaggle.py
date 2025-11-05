"""
Custom Dataset Loader for Spark-preprocessed Data on Kaggle
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

class SparkPreprocessedDataset(Dataset):
    """
    Load preprocessed data from Spark Parquet files
    Compatible with Bayes-PFL training
    """
    
    def __init__(self, parquet_path, mode='train', image_size=224):
        """
        Args:
            parquet_path: Path to train/test_features.parquet
            mode: 'train' or 'test'
            image_size: Image size (should match preprocessing: 224)
        """
        self.mode = mode
        self.image_size = image_size
        
        print(f"📂 Loading {mode} data from {parquet_path}...")
        
        # Read Parquet
        df = pd.read_parquet(parquet_path)
        
        # Extract data
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()
        self.categories = df['category'].tolist()
        self.splits = df['split'].tolist()
        
        # Features: reshape from flattened array to (3, 224, 224)
        self.features = []
        for feat in df['features'].tolist():
            feat_array = np.array(feat, dtype=np.float32)
            feat_array = feat_array.reshape(3, self.image_size, self.image_size)
            self.features.append(feat_array)
        
        print(f"✅ Loaded {len(self.features)} {mode} samples")
        print(f"   Categories: {len(set(self.categories))}")
        print(f"   Normal: {sum(1 for l in self.labels if l == 0)}")
        print(f"   Anomaly: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get preprocessed features (already normalized in CHW format)
        image_tensor = torch.from_numpy(self.features[idx])
        
        # Metadata
        label = self.labels[idx]
        category = self.categories[idx]
        path = self.paths[idx]
        
        # Create item dict compatible with Bayes-PFL
        item = {
            'img': image_tensor,
            'label': label,
            'cls_name': [category],
            'anomaly': torch.tensor(label, dtype=torch.long),
            'img_path': [path],
            'img_mask': torch.zeros(1, self.image_size, self.image_size)  # Dummy mask
        }
        
        return item


def create_spark_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Create DataLoaders from Spark preprocessed data
    
    Args:
        data_dir: Directory containing train/test_features.parquet
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, test_loader, category_list
    """
    train_path = os.path.join(data_dir, 'train_features.parquet')
    test_path = os.path.join(data_dir, 'test_features.parquet')
    
    # Create datasets
    train_dataset = SparkPreprocessedDataset(train_path, mode='train')
    test_dataset = SparkPreprocessedDataset(test_path, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Bayes-PFL uses batch_size=1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get category list
    categories = sorted(list(set(train_dataset.categories)))
    
    return train_loader, test_loader, categories


# Test loading
if __name__ == "__main__":
    # Example for Kaggle
    data_dir = "/kaggle/input/bayes-pfl-preprocessed-features"
    
    train_loader, test_loader, categories = create_spark_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2
    )
    
    print(f"\n📊 DataLoader Info:")
    print(f"Categories: {categories}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"\n✅ Batch loaded successfully!")
        print(f"Image shape: {batch['img'].shape}")
        print(f"Labels: {batch['label']}")
        print(f"Categories: {batch['cls_name']}")
        break