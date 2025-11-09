"""Inference engine for skin lesion classification.
Implements image quality validation, test-time augmentation, confidence calibration,
and clinical decision support with uncertainty quantification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
import json
from PIL import Image

from config import get_config
from utils import (setup_device, load_checkpoint, preprocess_image_for_model,
                   check_image_quality, get_tta_transforms, calculate_confidence_metrics,
                   get_disease_info)
from model import SkinLesionClassifier

logger = logging.getLogger(__name__)
config = get_config()


class SkinLesionPredictor:
    """Production-ready inference engine with quality validation and TTA.
    
    Implements comprehensive model deployment pipeline with image validation,
    augmentation-based uncertainty quantification, and clinical confidence assessment.
    """
    
    def __init__(self, model_path: Path, device: Optional[torch.device] = None):
        self.device = device or setup_device(config.device)
        self.model_path = Path(model_path)
        
        # Load model
        self.model, self.class_names = self._load_model()
        self.model.eval()
        
        logger.info(f"Predictor initialized with {len(self.class_names)} classes")
    
    def _load_model(self) -> Tuple[torch.nn.Module, List[str]]:
        """Load trained model checkpoint and restore weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint (weights_only=False for compatibility with our custom checkpoint format)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get class names
        class_names = checkpoint.get('class_names', config.data.class_names)
        
        # Create model
        model = SkinLesionClassifier(
            num_classes=len(class_names),
            backbone=config.model.backbone,
            dropout=0.0  # No dropout during inference
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
        logger.info(f"Validation accuracy: {checkpoint.get('metrics', {}).get('val_acc', 'N/A')}")
        
        return model, class_names
    
    @torch.no_grad()
    def predict_single(self, image_path: Path,
                       use_tta: bool = True,
                       check_quality: bool = True) -> Dict:
        """
        Predict disease class for single image with comprehensive validation.
        
        Returns dict with:
            - predicted_class: Most likely class name
            - confidence: Confidence percentage
            - probabilities: All class probabilities
            - confidence_metrics: Detailed confidence analysis
            - quality_check: Image quality assessment
            - clinical_info: Disease information
        """
        image_path = Path(image_path)
        
        # Step 1: Quality validation
        quality_result = {'is_valid': True, 'message': 'Quality check passed'}
        if check_quality and config.inference.check_image_quality:
            is_valid, message = check_image_quality(image_path)
            quality_result = {'is_valid': is_valid, 'message': message}
            
            if not is_valid:
                logger.warning(f"Quality check failed: {message}")
                # Continue but flag the issue
        
        # Step 2: Preprocess image
        try:
            image_tensor, original_image = preprocess_image_for_model(
                image_path,
                target_size=config.model.img_size,
                enhance_quality=True
            )
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
        
        # Step 3: Get predictions
        if use_tta and config.inference.use_tta:
            probabilities, uncertainty = self._predict_with_tta(image_tensor)
        else:
            probabilities = self._predict_single_transform(image_tensor)
            uncertainty = {}
        
        # Step 4: Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx] * 100
        
        # Step 5: Calculate confidence metrics
        confidence_metrics = calculate_confidence_metrics(probabilities)
        confidence_metrics.update(uncertainty)
        
        # Step 6: Clinical assessment
        clinical_assessment = self._assess_clinical_confidence(
            probabilities, confidence_metrics
        )
        
        # Step 7: Get disease information
        disease_info = get_disease_info(predicted_class)
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {self.class_names[i]: float(probabilities[i]) * 100 
                            for i in range(len(self.class_names))},
            'confidence_metrics': confidence_metrics,
            'clinical_assessment': clinical_assessment,
            'disease_info': disease_info,
            'quality_check': quality_result,
            'image_path': str(image_path),
            'tta_enabled': use_tta and config.inference.use_tta
        }
    
    def _predict_single_transform(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Single forward pass prediction"""
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        return probabilities
    
    def _predict_with_tta(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        """
        Test-Time Augmentation for robust predictions.
        Returns (average_probabilities, uncertainty_metrics)
        """
        tta_transforms = get_tta_transforms()
        all_predictions = []
        
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for transform_fn in tta_transforms:
                augmented = transform_fn(image_batch)
                outputs = self.model(augmented)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                all_predictions.append(probs)
        
        # Average predictions
        all_predictions = np.array(all_predictions)
        avg_probabilities = np.mean(all_predictions, axis=0)
        
        # Calculate uncertainty from TTA
        uncertainty = {
            'tta_std': float(np.mean(np.std(all_predictions, axis=0))),
            'tta_max_std': float(np.max(np.std(all_predictions, axis=0))),
            'tta_variance': float(np.mean(np.var(all_predictions, axis=0))),
            'tta_samples': len(all_predictions)
        }
        
        return avg_probabilities, uncertainty
    
    def _assess_clinical_confidence(self, 
                                    probabilities: np.ndarray,
                                    confidence_metrics: Dict) -> Dict:
        """
        Assess clinical confidence level based on multiple factors.
        Addresses the accuracy issue by providing honest confidence assessment.
        """
        max_prob = confidence_metrics['max_prob']
        confidence_gap = confidence_metrics['confidence_gap']
        entropy = confidence_metrics['entropy']
        
        # Determine confidence level
        if max_prob >= config.inference.high_confidence_threshold and confidence_gap > 0.3:
            clinical_confidence = "High"
            reliability = "Reliable for screening purposes"
        elif max_prob >= config.inference.medium_confidence_threshold and confidence_gap > 0.15:
            clinical_confidence = "Moderate"
            reliability = "Consider additional evaluation"
        elif max_prob >= config.inference.low_confidence_threshold:
            clinical_confidence = "Low"
            reliability = "Professional examination strongly recommended"
        else:
            clinical_confidence = "Very Low"
            reliability = "Inconclusive - requires clinical evaluation"
        
        # Check for high uncertainty
        if entropy > config.inference.high_uncertainty_entropy:
            clinical_confidence += " (High Uncertainty)"
            reliability = "Multiple diagnoses possible - clinical correlation essential"
        
        # Determine urgency
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        
        high_risk_classes = ['mel', 'bcc', 'akiec']
        urgent_classes = ['mel']
        
        if predicted_class in urgent_classes and max_prob > 0.6:
            urgency_level = "URGENT - Immediate medical evaluation required"
        elif predicted_class in high_risk_classes and max_prob > 0.5:
            urgency_level = "HIGH - Prompt dermatological consultation recommended"
        elif clinical_confidence in ["Very Low", "Low"]:
            urgency_level = "MODERATE - Clinical correlation needed"
        else:
            urgency_level = "ROUTINE - Standard monitoring appropriate"
        
        # Generate differential diagnosis
        sorted_indices = np.argsort(probabilities)[::-1][:3]
        differential = []
        for rank, idx in enumerate(sorted_indices, 1):
            if probabilities[idx] > 0.05:  # Only include >5%
                differential.append({
                    'rank': rank,
                    'condition': self.class_names[idx],
                    'probability': float(probabilities[idx] * 100)
                })
        
        return {
            'clinical_confidence': clinical_confidence,
            'reliability': reliability,
            'urgency_level': urgency_level,
            'differential_diagnosis': differential,
            'assessment_notes': self._generate_assessment_notes(
                predicted_class, max_prob, entropy
            )
        }
    
    def _generate_assessment_notes(self, 
                                   predicted_class: str,
                                   confidence: float,
                                   entropy: float) -> str:
        """Generate clinical assessment notes"""
        notes = []
        
        if confidence < 0.7:
            notes.append("Model confidence is below optimal threshold.")
            notes.append("Consider repeat imaging with better lighting/focus.")
        
        if entropy > 1.5:
            notes.append("High prediction uncertainty detected.")
            notes.append("Multiple conditions show similar probability.")
        
        if predicted_class in ['mel', 'bcc']:
            notes.append(f"Potential {predicted_class.upper()} detected.")
            notes.append("Urgent professional evaluation essential.")
        
        if not notes:
            notes.append("Prediction appears stable and reliable.")
            notes.append("Standard clinical protocols apply.")
        
        return " ".join(notes)
    
    def batch_predict(self, 
                     image_dir: Path,
                     use_tta: bool = True,
                     save_results: bool = True) -> List[Dict]:
        """Predict for all images in a directory"""
        image_dir = Path(image_dir)
        results = []
        
        # Find all images
        image_files = list(image_dir.glob('*.[jp][pn]g'))
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        for img_path in image_files:
            try:
                result = self.predict_single(img_path, use_tta=use_tta)
                results.append(result)
                
                logger.info(f"{img_path.name}: {result['predicted_class']} "
                          f"({result['confidence']:.2f}%)")
            
            except Exception as e:
                logger.error(f"Failed to process {img_path.name}: {e}")
        
        # Save results
        if save_results and config.save_predictions:
            output_path = config.output_dir / 'batch_predictions.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        return results


def main():
    """Test inference pipeline"""
    from utils import setup_logging
    setup_logging(config.log_level)
    
    model_path = config.inference.model_dir / config.inference.best_model_name
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Train the model first using: python model.py")
        return
    
    # Create predictor
    predictor = SkinLesionPredictor(model_path)
    
    # Test on sample image
    test_dirs = [Path('captured_images'), Path('test_images'), Path('samples')]
    test_image = None
    
    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.glob('*.[jp][pn]g'))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        logger.info(f"Testing on: {test_image}")
        result = predictor.predict_single(test_image, use_tta=True)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Image: {result['image_path']}")
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Clinical Confidence: {result['clinical_assessment']['clinical_confidence']}")
        print(f"Urgency: {result['clinical_assessment']['urgency_level']}")
        print(f"Quality Check: {result['quality_check']['message']}")
        print("\nTop 3 Predictions:")
        for item in result['clinical_assessment']['differential_diagnosis']:
            print(f"  {item['rank']}. {item['condition']}: {item['probability']:.2f}%")
        print("=" * 60)
    else:
        logger.warning("No test images found")


if __name__ == "__main__":
    main()
