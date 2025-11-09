"""Data preprocessing pipeline for skin lesion classification.
Handles dataset organization, stratified splitting, augmentation, and quality validation
with comprehensive logging and error handling.
"""

import pandas as pd
import shutil
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import logging
from typing import Tuple, List, Optional
from sklearn.model_selection import StratifiedShuffleSplit

from config import get_config
from utils import check_image_quality, setup_logging

logger = logging.getLogger(__name__)
config = get_config()


class SkinLesionDataset(Dataset):
    """PyTorch dataset for skin lesion classification with quality validation."""
    
    def __init__(self, 
                 data_dir: Path, 
                 transform: Optional[transforms.Compose] = None,
                 class_names: Optional[List[str]] = None,
                 validate_quality: bool = False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.validate_quality = validate_quality
        self.samples = []  # (image_path, label) tuples
        
        if class_names is None:
            self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self._load_samples()
        
    def _load_samples(self):
        """Load all samples with quality validation"""
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                continue
            
            for img_path in class_dir.glob('*.[jp][pn]g'):  # jpg, jpeg, png
                # Validate quality if required
                if self.validate_quality:
                    is_valid, _ = check_image_quality(img_path)
                    if not is_valid:
                        continue
                
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            fallback = torch.zeros(3, config.model.img_size[0], config.model.img_size[1])
            return fallback, label


def create_transforms(mode: str = 'train') -> transforms.Compose:
    """Create augmentation pipeline"""
    img_size = config.model.img_size
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(config.model.rotation_range),
            transforms.RandomHorizontalFlip(p=config.model.horizontal_flip_prob),
            transforms.RandomVerticalFlip(p=config.model.vertical_flip_prob),
            transforms.ColorJitter(
                brightness=config.model.brightness_range,
                contrast=config.model.contrast_range,
                saturation=(0.9, 1.1),
                hue=0.05
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def prepare_ham10000_data() -> Path:
    """Merge and organize HAM10000 dataset into class-stratified directory structure."""
    logger.info("Preparing HAM10000 dataset...")
    
    # Merge parts
    merged_dir = config.data.merged_dir
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    parts = [config.data.raw_ham10000_part1, config.data.raw_ham10000_part2]
    total = 0
    
    for part_dir in parts:
        if part_dir.exists():
            for img in part_dir.glob("*.jpg"):
                try:
                    Image.open(img).verify()  # Validate
                    shutil.copy(img, merged_dir / img.name)
                    total += 1
                except:
                    logger.warning(f"Skipping corrupt: {img.name}")
    
    logger.info(f"Merged {total} images")
    
    # Organize by class
    processed_dir = config.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = pd.read_csv(config.data.metadata_path)
    metadata = metadata.dropna(subset=['image_id', 'dx'])
    
    for class_name in metadata['dx'].unique():
        (processed_dir / class_name).mkdir(exist_ok=True)
    
    organized = 0
    for _, row in metadata.iterrows():
        src = merged_dir / f"{row['image_id']}.jpg"
        if src.exists():
            dst = processed_dir / row['dx'] / f"{row['image_id']}.jpg"
            shutil.copy(src, dst)
            organized += 1
    
    logger.info(f"Organized {organized} images by class")
    return processed_dir


def add_normal_skin_samples(processed_dir: Path):
    """Add healthy skin samples"""
    normal_dir = processed_dir / 'normal'
    normal_dir.mkdir(exist_ok=True)
    
    normal_sources = config.data.normal_skin_dir
    if not normal_sources.exists():
        logger.warning("Normal skin dataset not found - skipping")
        return
    
    added = 0
    for img_path in normal_sources.rglob('*.[jp][pn]g'):
        try:
            Image.open(img_path).verify()
            dst = normal_dir / f"normal_{img_path.parent.name}_{img_path.name}"
            shutil.copy(img_path, dst)
            added += 1
        except:
            pass
    
    logger.info(f"Added {added} normal skin samples")


def create_data_loaders(data_dir: Path,
                        batch_size: int = None,
                        use_weighted_sampler: bool = True) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create optimized data loaders with stratified train/validation split.
    
    Applies class weights to address imbalance and uses weighted random sampling.
    Returns (train_loader, val_loader, class_names) tuple.
    """
    if batch_size is None:
        batch_size = config.model.batch_size
    
    # Create full dataset
    full_dataset = SkinLesionDataset(data_dir, transform=None, validate_quality=True)
    class_names = full_dataset.class_names
    
    # Stratified split
    labels = [label for _, label in full_dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.data.val_split, random_state=config.data.random_seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    # Create subsets with appropriate transforms
    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples = [full_dataset.samples[i] for i in val_idx]
    
    train_dataset = SkinLesionDataset.__new__(SkinLesionDataset)
    train_dataset.samples = train_samples
    train_dataset.class_names = class_names
    train_dataset.class_to_idx = full_dataset.class_to_idx
    train_dataset.transform = create_transforms('train')
    train_dataset.validate_quality = False
    
    val_dataset = SkinLesionDataset.__new__(SkinLesionDataset)
    val_dataset.samples = val_samples
    val_dataset.class_names = class_names
    val_dataset.class_to_idx = full_dataset.class_to_idx
    val_dataset.transform = create_transforms('val')
    val_dataset.validate_quality = False
    
    # Weighted sampler for training
    if use_weighted_sampler:
        train_labels = [label for _, label in train_samples]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info(f"Classes: {class_names}")
    
    return train_loader, val_loader, class_names


def main():
    """Execute data preprocessing pipeline including dataset organization and validation."""
    setup_logging(config.log_level, config.log_dir / 'preprocessing.log')
    
    logger.info("Initiating data preprocessing pipeline.")
    
    # Step 1: Prepare HAM10000 data
    processed_dir = prepare_ham10000_data()
    
    # Step 2: Add normal skin samples
    add_normal_skin_samples(processed_dir)
    
    # Step 3: Create and validate dataloaders
    train_loader, val_loader, class_names = create_data_loaders(processed_dir)
    
    # Display statistics
    logger.info("=" * 60)
    logger.info(f"Preprocessing complete!")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Classes: {', '.join(class_names)}")
    logger.info(f"Data ready for training!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
