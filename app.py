"""Main application entry point for skin lesion classification system.
Provides comprehensive command-line interface for model training, data preprocessing,
and clinical inference with integrated diagnostic assessment capabilities.
"""

import argparse
import sys
from pathlib import Path
import logging

from config import get_config
from utils import setup_logging, setup_device, get_disease_info
from data_pipeline import main as preprocess_data
from model import main as train_model
from predictor import SkinLesionPredictor
from diagnostic_report_generator import generate_diagnostic_report

logger = logging.getLogger(__name__)
config = get_config()


def display_system_info():
    """Display system and configuration information."""
    import torch
    
    print("\n" + "=" * 70)
    print("SKIN LESION CLASSIFICATION SYSTEM")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model.backbone}")
    print(f"  Image Size: {config.model.img_size}")
    print(f"  Batch Size: {config.model.batch_size}")
    print(f"  Classes: {len(config.data.class_names)}")
    print("=" * 70 + "\n")


def cmd_preprocess(args):
    """Preprocess and organize dataset for training."""
    logger.info("Initiating data preprocessing.")
    preprocess_data()
    logger.info("Data preprocessing completed.")


def cmd_train(args):
    """Train classifier model on preprocessed dataset."""
    logger.info("Initiating model training.")
    
    if not config.data.processed_dir.exists():
        logger.error("Processed data directory not found.")
        logger.error("Execute data preprocessing: python app.py preprocess")
        sys.exit(1)
    
    train_model()
    logger.info("Model training completed.")


def cmd_predict(args):
    """Classify skin lesion from image file or directory."""
    model_path = config.inference.model_dir / config.inference.best_model_name
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Train the model first: python app.py train")
        sys.exit(1)
    
    # Create predictor
    predictor = SkinLesionPredictor(model_path)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image prediction
        logger.info(f"Predicting on single image: {input_path}")
        result = predictor.predict_single(
            input_path,
            use_tta=not args.no_tta,
            check_quality=not args.skip_quality_check
        )
        
        display_prediction_result(result, args.verbose)
        
        # Save result if requested
        if args.save:
            output_file = config.output_dir / f"{input_path.stem}_prediction.json"
            import json
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"[SUCCESS] Result saved to: {output_file}")
        
        # Generate diagnostic report if requested
        if args.report:
            logger.info("Generating diagnostic PDF report...")
            
            # Convert probabilities dict to numpy array in class order
            import numpy as np
            predictions_array = np.array([
                result['probabilities'][class_name] 
                for class_name in config.data.class_names
            ])
            
            report_path = generate_diagnostic_report(
                image_path=str(input_path),
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                predictions=predictions_array,
                class_names=config.data.class_names,
                clinical_assessment=result.get('clinical_assessment')
            )
            if report_path:
                logger.info(f"[SUCCESS] Diagnostic report saved to: {report_path}")
            else:
                logger.warning("[FAIL] Report generation failed")
    
    elif input_path.is_dir():
        # Batch prediction
        logger.info(f"Batch prediction on directory: {input_path}")
        results = predictor.batch_predict(
            input_path,
            use_tta=not args.no_tta,
            save_results=args.save
        )
        
        # Display summary
        display_batch_summary(results)
    
    else:
        logger.error(f"Invalid path: {input_path}")
        sys.exit(1)


def display_prediction_result(result: dict, verbose: bool = False):
    """Display prediction result with clinical assessment information."""
    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    
    print(f"\nPREDICTION RESULT:")
    print(f"  Class: {result['predicted_class'].upper()}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    
    disease = result['disease_info']
    print(f"\nCLINICAL INFORMATION:")
    print(f"  Name: {disease['full_name']}")
    print(f"  Risk Level: {disease['severity']}")
    print(f"  Description: {disease['description']}")
    
    assessment = result['clinical_assessment']
    print(f"\nCLINICAL ASSESSMENT:")
    print(f"  Confidence Level: {assessment['clinical_confidence']}")
    print(f"  Reliability: {assessment['reliability']}")
    print(f"  Urgency: {assessment['urgency_level']}")
    
    quality = result['quality_check']
    quality_status = "PASS" if quality['is_valid'] else "FAIL"
    print(f"\nQUALITY CHECK: {quality_status}")
    print(f"  {quality['message']}")
    
    tta_status = "Enabled" if result['tta_enabled'] else "Disabled"
    print(f"\nTEST-TIME AUGMENTATION: {tta_status}")
    
    # Verbose output
    if verbose:
        print(f"\nDIFFERENTIAL DIAGNOSIS:")
        for item in assessment['differential_diagnosis']:
            print(f"  {item['rank']}. {item['condition'].upper()}: {item['probability']:.2f}%")
        
        print(f"\nCONFIDENCE METRICS:")
        metrics = result['confidence_metrics']
        print(f"  Entropy: {metrics.get('entropy', 0):.4f}")
        print(f"  Confidence Gap: {metrics.get('confidence_gap', 0):.4f}")
        print(f"  Top-3 Sum: {metrics.get('top3_sum', 0):.4f}")
        
        if 'tta_std' in metrics:
            print(f"  TTA Std Dev: {metrics.get('tta_std', 0):.4f}")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in disease['recommendations']:
            print(f"  • {rec}")
    
    print("\n" + "=" * 70 + "\n")


def display_batch_summary(results: list):
    """Display summary of batch classification results."""
    print("\n" + "=" * 70)
    print("BATCH PREDICTION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Images Processed: {len(results)}")
    
    class_counts = {}
    avg_confidence = 0
    high_urgency = 0
    
    for result in results:
        pred_class = result['predicted_class']
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        avg_confidence += result['confidence']
        
        if 'URGENT' in result['clinical_assessment']['urgency_level']:
            high_urgency += 1
    
    avg_confidence /= len(results)
    
    print(f"\nCLASS DISTRIBUTION:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {class_name.upper()}: {count} ({percentage:.1f}%)")
    
    print(f"\nAGGREGATE STATISTICS:")
    print(f"  Average Confidence: {avg_confidence:.2f}%")
    print(f"  High Priority Cases: {high_urgency}")
    
    print("\n" + "=" * 70 + "\n")


def cmd_info(args):
    """Display disease classification reference information."""
    if args.disease:
        disease_info = get_disease_info(args.disease)
        
        print("\n" + "=" * 70)
        print(f"{disease_info['full_name'].upper()}")
        print("=" * 70)
        print(f"\nCode: {args.disease.upper()}")
        print(f"Risk Level: {disease_info['severity']}")
        print(f"\nDescription:")
        print(f"  {disease_info['description']}")
        print(f"\nRecommendations:")
        for rec in disease_info['recommendations']:
            print(f"  • {rec}")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("SUPPORTED DISEASE CLASSES")
        print("=" * 70 + "\n")
        
        for class_name in config.data.class_names:
            disease_info = get_disease_info(class_name)
            print(f"{class_name.upper():8} - {disease_info['full_name']}")
            print(f"         {disease_info['severity']}")
            print()
        
        print("=" * 70 + "\n")


def main():
    """Main command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description='Skin Lesion Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  python app.py preprocess
  
  # Train model
  python app.py train
  
  # Predict single image
  python app.py predict image.jpg
  python app.py predict image.jpg --verbose
  python app.py predict image.jpg --no-tta --save
  python app.py predict image.jpg --report
  
  # Batch predict
  python app.py predict ./images/ --save
  
  # View disease information
  python app.py info
  python app.py info --disease mel
        """
    )
    
    parser.add_argument('--system-info', action='store_true',
                       help='Display system information')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Preprocess command
    parser_preprocess = subparsers.add_parser('preprocess',
                                              help='Preprocess and organize dataset')
    
    # Train command
    parser_train = subparsers.add_parser('train',
                                         help='Train the model')
    
    # Predict command
    parser_predict = subparsers.add_parser('predict',
                                           help='Classify skin lesion from image(s)')
    parser_predict.add_argument('input', type=str,
                               help='Image file or directory path')
    parser_predict.add_argument('--no-tta', action='store_true',
                               help='Disable test-time augmentation')
    parser_predict.add_argument('--skip-quality-check', action='store_true',
                               help='Skip image quality validation')
    parser_predict.add_argument('--save', action='store_true',
                               help='Save prediction results to file')
    parser_predict.add_argument('--report', action='store_true',
                               help='Generate diagnostic PDF report')
    parser_predict.add_argument('--verbose', '-v', action='store_true',
                               help='Show detailed prediction information')
    
    # Info command
    parser_info = subparsers.add_parser('info',
                                        help='Display disease class information')
    parser_info.add_argument('--disease', type=str,
                            help='Specific disease code (e.g., mel, bcc)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(config.log_level, config.log_dir / 'app.log')
    
    # Display system info if requested
    if args.system_info or args.command is None:
        display_system_info()
    
    # Execute command
    if args.command == 'preprocess':
        cmd_preprocess(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
