"""Utility functions for skin disease classification system.
Provides image quality validation, preprocessing, model checkpointing, and
metrics computation for clinical decision support.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[Path] = None):
    """Configure logging system with console and optional file output."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create handlers with UTF-8 encoding for Windows compatibility
    import sys
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    
    handlers = [stream_handler]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )


def setup_device(device_preference: str = 'auto') -> torch.device:
    """Setup computation device with automatic selection or manual preference.
    
    Validates device availability and configures CUDA optimizations if GPU is present.
    """
    if device_preference == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_preference)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        torch.backends.cudnn.benchmark = True
    
    return device


def check_image_quality(image_path: Path, 
                        min_resolution: Tuple[int, int] = (100, 100),
                        max_blur_threshold: float = 100.0) -> Tuple[bool, str]:
    """Validate image quality prior to classification.
    
    Checks image resolution, blur level via Laplacian variance, and exposure levels.
    Returns (is_valid, message) tuple indicating quality assessment result.
    """
    try:
        # Check if file exists and is readable
        if not image_path.exists():
            return False, f"Image file not found: {image_path}"
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Unable to read image file"
        
        # Check resolution
        height, width = img.shape[:2]
        if width < min_resolution[0] or height < min_resolution[1]:
            return False, f"Image resolution too low: {width}x{height}. Minimum: {min_resolution}"
        
        # Check for blur using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < max_blur_threshold:
            return False, f"Image too blurry (variance: {laplacian_var:.2f}). Try capturing again with better focus."
        
        # Check for extreme brightness/darkness
        mean_brightness = gray.mean()
        if mean_brightness < 20:
            return False, "Image too dark. Ensure adequate lighting."
        if mean_brightness > 235:
            return False, "Image overexposed. Reduce lighting or adjust camera settings."
        
        return True, "Image quality acceptable"
        
    except Exception as e:
        return False, f"Error checking image quality: {str(e)}"


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality using adaptive preprocessing.
    
    Applies CLAHE for contrast enhancement, bilateral filtering for noise reduction,
    and normalizes lighting conditions to improve feature extraction.
    """
    # Convert to PIL for processing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Auto-enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Convert back to numpy/opencv format
    enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Reduce noise with bilateral filter
    enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
    
    return enhanced


def preprocess_image_for_model(image_path: Path,
                                target_size: Tuple[int, int] = (224, 224),
                                enhance_quality: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess image for model inference.
    
    Handles image loading, quality enhancement, and tensor conversion.
    Returns (normalized_tensor, original_array) for inference and visualization.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    
    # Enhance quality if requested
    if enhance_quality:
        image = enhance_image_quality(image)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
    
    return tensor, image_rgb


def calculate_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy of probability distribution.
    
    Measures prediction uncertainty; higher values indicate greater uncertainty.
    """
    # Avoid log(0)
    probs = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def calculate_confidence_metrics(probabilities: np.ndarray) -> dict:
    """Compute multi-dimensional confidence metrics for prediction assessment.
    
    Evaluates model confidence through entropy, probability gap, and Gini coefficient.
    """
    sorted_probs = np.sort(probabilities)[::-1]
    
    metrics = {
        'max_prob': float(sorted_probs[0]),
        'second_max_prob': float(sorted_probs[1] if len(sorted_probs) > 1 else 0),
        'confidence_gap': float(sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]),
        'entropy': float(calculate_entropy(probabilities)),
        'gini_coefficient': float(1 - np.sum(probabilities ** 2)),
        'top3_sum': float(np.sum(sorted_probs[:3]))
    }
    
    return metrics


def apply_temperature_scaling(logits: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    """Apply temperature scaling to calibrate prediction confidence.
    
    Temperature values greater than 1 reduce confidence (increase entropy);
    values less than 1 sharpen predictions (decrease entropy).
    """
    return torch.softmax(logits / temperature, dim=-1)


def get_tta_transforms() -> List[callable]:
    """Generate test-time augmentation transformation ensemble.
    
    Returns nine geometric and photometric transformations for uncertainty quantification
    via prediction averaging across augmented versions.
    """
    transforms = [
        # Original
        lambda x: x,
        # Horizontal flip
        lambda x: torch.flip(x, dims=[3]),
        # Vertical flip
        lambda x: torch.flip(x, dims=[2]),
        # Both flips
        lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3]),
        # Rotations
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=2, dims=[2, 3]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),
        # Brightness adjustment (slight)
        lambda x: torch.clamp(x * 1.05, -3, 3),
        lambda x: torch.clamp(x * 0.95, -3, 3),
    ]
    
    return transforms


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: dict,
                   save_path: Path,
                   additional_info: Optional[dict] = None):
    """Save training checkpoint with model state and metrics."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path: Path,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: torch.device = None) -> dict:
    """Load saved checkpoint and restore model and optimizer state."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


class AverageMeter:
    """Computes running average and current values for metrics tracking."""
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Monitor metric and halt training when no improvement is observed.
    
    Implements patience-based early stopping mechanism to prevent overfitting
    and reduce unnecessary training iterations.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.early_stop


def get_class_names() -> List[str]:
    """Return list of supported disease classification categories."""
    return ['akiec', 'bcc', 'bkl', 'df', 'mel', 'normal', 'nv', 'vasc']


def get_disease_info(class_name: str) -> dict:
    """Retrieve clinical information and recommendations for disease classification."""
    disease_database = {
        'akiec': {
            'full_name': 'Actinic Keratoses and Intraepithelial Carcinoma',
            'severity': 'Medium to High Risk - Precancerous',
            'description': 'Sun-damaged skin lesions that can progress to squamous cell carcinoma if untreated.',
            'recommendations': [
                'Urgent dermatological evaluation within 2-4 weeks',
                'Consider cryotherapy, topical chemotherapy, or photodynamic therapy',
                'Aggressive sun protection and regular monitoring',
                'Full body skin examination recommended'
            ]
        },
        'bcc': {
            'full_name': 'Basal Cell Carcinoma',
            'severity': 'High Risk - Most Common Skin Cancer',
            'description': 'Slow-growing cancer from basal cells. Rarely metastasizes but can cause local tissue damage.',
            'recommendations': [
                'Urgent dermatology/oncology referral',
                'Surgical excision (Mohs surgery preferred for facial lesions)',
                'Complete skin examination for additional lesions',
                'Regular follow-up every 6-12 months'
            ]
        },
        'bkl': {
            'full_name': 'Benign Keratosis (Seborrheic/Solar Lentigo)',
            'severity': 'Low Risk - Benign',
            'description': 'Common benign skin growths, not cancerous. Often age-related.',
            'recommendations': [
                'Routine monitoring for changes',
                'Consider removal if cosmetically bothersome',
                'Annual dermatology screening if multiple lesions',
                'Document with photography for comparison'
            ]
        },
        'df': {
            'full_name': 'Dermatofibroma',
            'severity': 'Low Risk - Benign',
            'description': 'Benign fibrous nodule, often following minor trauma. Completely harmless.',
            'recommendations': [
                'No treatment typically required',
                'Surgical excision if symptomatic or rapidly changing',
                'Monitor for any unusual changes',
                'Reassure patient of benign nature'
            ]
        },
        'mel': {
            'full_name': 'Melanoma',
            'severity': 'VERY HIGH RISK - Aggressive Cancer',
            'description': 'Most dangerous skin cancer with high metastatic potential. Early detection critical.',
            'recommendations': [
                'URGENT: Immediate oncology referral (within 24-48 hours)',
                'Excisional biopsy required for staging',
                'Sentinel lymph node biopsy if indicated',
                'Full body PET/CT if metastasis suspected',
                'Discuss treatment options: surgery, immunotherapy, targeted therapy'
            ]
        },
        'normal': {
            'full_name': 'Normal Healthy Skin',
            'severity': 'No Risk',
            'description': 'Healthy skin tissue without pathological changes.',
            'recommendations': [
                'Continue monthly self-examinations (ABCDE method)',
                'Annual dermatology screening if family history',
                'Sun protection: SPF 30+ daily, protective clothing',
                'Avoid tanning beds and excessive sun exposure'
            ]
        },
        'nv': {
            'full_name': 'Melanocytic Nevus (Mole)',
            'severity': 'Low Risk - Usually Benign',
            'description': 'Common benign mole. Monitor for changes indicating malignant transformation.',
            'recommendations': [
                'Monitor for ABCDE changes: Asymmetry, Border, Color, Diameter, Evolution',
                'Photograph for comparison over time',
                'Consider removal if atypical features or patient concern',
                'Annual skin check if multiple atypical nevi'
            ]
        },
        'vasc': {
            'full_name': 'Vascular Lesions (Hemangioma/Angiokeratoma)',
            'severity': 'Low Risk - Benign',
            'description': 'Benign proliferation of blood vessels. Cosmetic concern more than medical.',
            'recommendations': [
                'Monitor for bleeding or rapid growth',
                'Laser therapy if cosmetically concerning',
                'Consider removal if frequently traumatized',
                'Routine monitoring sufficient for most cases'
            ]
        }
    }
    
    return disease_database.get(class_name, {
        'full_name': class_name.upper(),
        'severity': 'Unknown',
        'description': 'No information available for this classification.',
        'recommendations': ['Consult healthcare professional for evaluation']
    })
