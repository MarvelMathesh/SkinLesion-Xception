"""Centralized configuration management for skin disease classification.
Defines all training hyperparameters, model architecture settings, inference parameters,
and system paths in a single configuration object.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ModelConfig:
    """Model architecture and training hyperparameters"""
    # Model architecture
    backbone: str = 'xception'
    num_classes: int = 8
    pretrained: bool = True
    dropout_rate: float = 0.4
    
    # Training hyperparameters
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 10
    
    # Optimizer and scheduler
    optimizer: str = 'adamw'
    scheduler: str = 'reduce_on_plateau'
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Data augmentation
    rotation_range: int = 20
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3
    
    # Class weighting
    use_class_weights: bool = True
    class_weight_max: float = 3.0
    
    # Mixed precision training
    use_amp: bool = True
    gradient_clip_norm: float = 1.0


@dataclass
class DataConfig:
    """Data paths and preprocessing settings"""
    # Paths
    base_dir: Path = Path("data")
    raw_ham10000_part1: Path = Path("skin-cancer-mnist-ham10000/HAM10000_images_part_1")
    raw_ham10000_part2: Path = Path("skin-cancer-mnist-ham10000/HAM10000_images_part_2")
    metadata_path: Path = Path("skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
    normal_skin_dir: Path = Path("Oily-Dry-Skin-Types")
    
    merged_dir: Path = Path("data/HAM10000_merged")
    processed_dir: Path = Path("data/processed")
    
    # Data split
    train_split: float = 0.8
    val_split: float = 0.2
    random_seed: int = 42
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: [
        'akiec', 'bcc', 'bkl', 'df', 'mel', 'normal', 'nv', 'vasc'
    ])
    
    # Data loader settings
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class InferenceConfig:
    """Inference and prediction settings"""
    # Model paths
    model_dir: Path = Path("models")
    best_model_name: str = "best_model.pth"
    
    # Test-Time Augmentation
    use_tta: bool = True
    tta_transforms: int = 8  # Number of TTA augmentations
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.70
    low_confidence_threshold: float = 0.50
    
    # Uncertainty thresholds
    high_uncertainty_entropy: float = 1.5
    
    # Image quality checks
    check_image_quality: bool = True
    min_image_resolution: Tuple[int, int] = (100, 100)
    max_blur_threshold: float = 100.0  # Laplacian variance threshold


@dataclass
class ReportConfig:
    """Diagnostic report generation settings"""
    # Report paths
    reports_dir: Path = Path("reports")
    
    # Gemini AI
    gemini_api_key: str = "AIzaSyB8UcVhtTEDeFnjEKr0Whpa18EcZ1UpiWE"
    gemini_model: str = "gemini-2.0-flash-exp"
    
    # PDF settings
    page_size: str = "A4"
    include_ai_analysis: bool = True
    include_patient_info: bool = True


@dataclass
class FirebaseConfig:
    """Firebase/Firestore configuration"""
    # Firebase settings
    project_id: str = "vehnicate-38c38"
    service_account_path: Path = Path("vehnicate-38c38-firebase-adminsdk-fbsvc-86982a906b.json")
    
    # Collections
    collection_patients: str = "patients"
    collection_diagnoses: str = "diagnoses"
    collection_reports: str = "reports"
    collection_sessions: str = "clinical_sessions"
    
    # Storage settings
    store_to_cloud: bool = True


@dataclass
class SystemConfig:
    """Overall system configuration"""
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    
    # Device settings
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    cuda_benchmark: bool = True
    deterministic: bool = False
    
    # Logging
    log_level: str = 'INFO'
    save_predictions: bool = True
    save_visualizations: bool = True
    
    # Output paths
    output_dir: Path = Path("output")
    log_dir: Path = Path("logs")
    
    def __post_init__(self):
        """Create necessary directories"""
        self.data.base_dir.mkdir(parents=True, exist_ok=True)
        self.data.processed_dir.mkdir(parents=True, exist_ok=True)
        self.inference.model_dir.mkdir(parents=True, exist_ok=True)
        self.report.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Retrieve global configuration instance."""
    return config


def update_config(**kwargs):
    """Update configuration values dynamically."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            for section in ['model', 'data', 'inference', 'report', 'firebase']:
                section_config = getattr(config, section)
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
                    break
