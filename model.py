"""Model architecture and training pipeline for skin lesion classification.
Implements Xception backbone with optimized training procedure including mixed precision
training, gradient accumulation, and automated checkpoint management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
import timm
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
from typing import Tuple, Dict, Optional

from config import get_config
from utils import (setup_logging, setup_device, save_checkpoint, load_checkpoint,
                   AverageMeter, EarlyStopping, apply_temperature_scaling)
from data_pipeline import create_data_loaders

logger = logging.getLogger(__name__)
config = get_config()


class SkinLesionClassifier(nn.Module):
    """Deep neural network classifier with regularization and temperature scaling.
    
    Utilizes Xception pretrained backbone with optimized fully-connected head
    and batch normalization for improved generalization.
    """
    
    def __init__(self, num_classes: int = 8, backbone: str = 'xception', dropout: float = 0.4):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        
        # Improved classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.75),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights with proper initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def predict_proba(self, x):
        """Compute calibrated probability distributions via temperature scaling."""
        logits = self.forward(x)
        return apply_temperature_scaling(logits, self.temperature)


def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Initialize optimizer with differential learning rates per module."""
    # Different learning rates for backbone and classifier
    params = [
        {'params': model.backbone.parameters(), 'lr': config.model.learning_rate * 0.1},
        {'params': model.classifier.parameters(), 'lr': config.model.learning_rate}
    ]
    
    if config.model.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, weight_decay=config.model.weight_decay)
    elif config.model.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=config.model.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.model.optimizer}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer):
    """Create learning rate scheduler for dynamic adjustment during training."""
    if config.model.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.model.scheduler_factor,
            patience=config.model.scheduler_patience,
            min_lr=config.model.min_lr
        )
    elif config.model.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.model.epochs,
            eta_min=config.model.min_lr
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(model: nn.Module,
                   train_loader,
                   criterion,
                   optimizer,
                   scaler,
                   device: torch.device,
                   epoch: int) -> Dict:
    """Execute single training epoch with mixed precision and gradient accumulation."""
    model.train()
    
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        # Mixed precision training
        if config.model.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.model.gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.model.gradient_clip_norm)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0) * 100
        
        losses.update(loss.item(), targets.size(0))
        top1.update(acc, targets.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%'
        })
    
    return {'loss': losses.avg, 'acc': top1.avg}


@torch.no_grad()
def validate(model: nn.Module,
            val_loader,
            criterion,
            device: torch.device,
            epoch: int) -> Dict:
    """Validate model with comprehensive metrics including per-class accuracy."""
    model.eval()
    
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        if config.model.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        losses.update(loss.item(), targets.size(0))
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0) * 100
        top1.update(acc, targets.size(0))
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%'
        })
    
    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Per-class accuracy
    cm = confusion_matrix(all_targets, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'loss': losses.avg,
        'acc': top1.avg,
        'preds': all_preds,
        'targets': all_targets,
        'probs': all_probs,
        'per_class_acc': per_class_acc
    }


def train_model(model: nn.Module,
               train_loader,
               val_loader,
               class_names: list,
               device: torch.device) -> Dict:
    """Complete training loop with best practices"""
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    scaler = GradScaler(enabled=config.model.use_amp)
    early_stopping = EarlyStopping(patience=config.model.early_stopping_patience, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    logger.info("Initiating model training procedure.")
    
    for epoch in range(config.model.epochs):
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['acc'])
            else:
                scheduler.step()
        
        # Save best model
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(
                model, optimizer, epoch,
                {'val_acc': val_metrics['acc'], 'val_loss': val_metrics['loss']},
                config.inference.model_dir / config.inference.best_model_name,
                {'class_names': class_names, 'config': config}
            )
            logger.info(f"Model checkpoint saved. Validation accuracy: {val_metrics['acc']:.2f}%")
        
        # Early stopping
        if early_stopping(val_metrics['acc']):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final training summary
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    return history


def main():
    """Execute complete training pipeline including data loading and model optimization."""
    # Setup
    setup_logging(config.log_level, config.log_dir / 'training.log')
    device = setup_device(config.device)
    
    logger.info("Initializing training pipeline...")
    
    # Load data
    logger.info("Loading data...")
    processed_dir = config.data.processed_dir
    if not processed_dir.exists():
        logger.error("Processed data not found. Run data preprocessing first.")
        return
    
    train_loader, val_loader, class_names = create_data_loaders(processed_dir)
    
    # Create model
    logger.info("Creating model...")
    model = SkinLesionClassifier(
        num_classes=len(class_names),
        backbone=config.model.backbone,
        dropout=config.model.dropout_rate
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    history = train_model(model, train_loader, val_loader, class_names, device)
    
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
