"""Firebase Firestore integration for clinical data persistence.
Provides cloud storage and retrieval of diagnostic results, clinical assessments,
and patient information with comprehensive session management.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, List
import uuid
import logging

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    print("Firebase Admin SDK not available. Install with: pip install firebase-admin")
    FIREBASE_AVAILABLE = False

class FirebaseManager:
    """Firestore operations manager for clinical data with centralized configuration.
    
    Uses config.firebase for all settings including service account path, project ID,
    and collection names to maintain consistency with system architecture.
    """
    
    def __init__(self, service_account_path: Optional[str] = None):
        """Initialize Firebase connection using centralized configuration."""
        self.db = None
        self.app = None
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase integration disabled - SDK not available")
            return
        
        try:
            # Use config path if not explicitly provided
            account_path = service_account_path or str(config.firebase.service_account_path)
            
            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                if account_path and os.path.exists(account_path):
                    cred = credentials.Certificate(account_path)
                    self.app = firebase_admin.initialize_app(cred, {
                        'projectId': config.firebase.project_id
                    })
                    logger.info(f"Firebase initialized with project: {config.firebase.project_id}")
                else:
                    try:
                        self.app = firebase_admin.initialize_app()
                        logger.info("Firebase initialized with default credentials")
                    except Exception as e:
                        logger.error(f"Firebase initialization failed: {e}")
                        logger.info("Configure service account in config.py")
                        return
            else:
                self.app = firebase_admin.get_app()
            
            self.db = firestore.client()
            logger.info("Firebase Firestore connected successfully")
            
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            self.db = None
    
    def is_connected(self) -> bool:
        """Check if Firebase is properly connected."""
        return self.db is not None
    
    def create_collections_structure(self):
        """Create initial collection structure with sample documents."""
        if not self.is_connected():
            print("Firebase not connected. Cannot create collections.")
            return False
        
        try:
            # Use collection names from config
            collections_structure = {
                config.firebase.collection_patients: {
                    'patient_id': 'unique_patient_identifier',
                    'name': 'Patient Name',
                    'dob': '1990-01-01',
                    'gender': 'M/F/Other',
                    'contact_info': {
                        'email': 'patient@example.com',
                        'phone': '+1234567890'
                    },
                    'medical_history': [],
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                },
                config.firebase.collection_diagnoses: {
                    'diagnosis_id': 'unique_diagnosis_identifier',
                    'patient_id': 'reference_to_patient',
                    'image_path': 'path/to/lesion/image.jpg',
                    'predicted_class': 'normal',
                    'confidence_score': 85.5,
                    'clinical_assessment': {
                        'clinical_confidence': 'High',
                        'urgency_level': 'ROUTINE',
                        'differential_diagnosis': [],
                        'recommendations': []
                    },
                    'ai_analysis': 'Detailed AI analysis text',
                    'model_info': {
                        'model_version': 'Xception-98',
                        'validation_accuracy': 98.0,
                        'tta_enabled': True
                    },
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                },
                config.firebase.collection_reports: {
                    'report_id': 'unique_report_identifier',
                    'diagnosis_id': 'reference_to_diagnosis',
                    'patient_id': 'reference_to_patient',
                    'report_path': 'path/to/generated/report.pdf',
                    'report_type': 'diagnostic_report',
                    'generated_at': datetime.now(),
                    'status': 'completed'
                },
                config.firebase.collection_sessions: {
                    'session_id': 'unique_session_identifier',
                    'patient_id': 'reference_to_patient',
                    'diagnoses': ['diagnosis_id_1', 'diagnosis_id_2'],
                    'session_date': datetime.now(),
                    'clinician_notes': 'Additional clinical observations',
                    'follow_up_required': True,
                    'follow_up_date': None
                }
            }
            
            # Create sample documents for each collection
            for collection_name, sample_data in collections_structure.items():
                doc_ref = self.db.collection(collection_name).document('sample_' + collection_name)
                doc_ref.set(sample_data)
                logger.info(f"Created sample document in {collection_name} collection")
            
            logger.info("Firebase collections structure created successfully.")
            return True
            
        except Exception as e:
            print(f"Error creating collections structure: {e}")
            return False
    
    def save_patient_info(self, patient_info: Dict) -> str:
        """Save patient information to Firestore."""
        if not self.is_connected():
            return None
        
        try:
            # Generate unique patient ID if not provided
            patient_id = patient_info.get('patient_id', str(uuid.uuid4()))
            
            patient_data = {
                'patient_id': patient_id,
                'name': patient_info.get('name', 'Unknown'),
                'dob': patient_info.get('dob', 'N/A'),
                'gender': patient_info.get('gender', 'N/A'),
                'contact_info': patient_info.get('contact_info', {}),
                'medical_history': patient_info.get('medical_history', []),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Save to patients collection
            doc_ref = self.db.collection(config.firebase.collection_patients).document(patient_id)
            doc_ref.set(patient_data, merge=True)
            
            logger.info(f"Patient information saved with ID: {patient_id}")
            return patient_id
            
        except Exception as e:
            print(f"Error saving patient info: {e}")
            return None
    
    def save_diagnosis(self, diagnosis_data: Dict) -> str:
        """Save diagnosis and clinical assessment to Firestore."""
        if not self.is_connected():
            return None
        
        try:
            # Generate unique diagnosis ID
            diagnosis_id = str(uuid.uuid4())
            
            # Prepare diagnosis document
            diagnosis_doc = {
                'diagnosis_id': diagnosis_id,
                'patient_id': diagnosis_data.get('patient_id', 'unknown'),
                'image_path': diagnosis_data.get('image_path', ''),
                'predicted_class': diagnosis_data.get('predicted_class', ''),
                'confidence_score': float(diagnosis_data.get('confidence_score', 0)),
                'all_predictions': diagnosis_data.get('all_predictions', {}),
                'clinical_assessment': diagnosis_data.get('clinical_assessment', {}),
                'ai_analysis': diagnosis_data.get('ai_analysis', ''),
                'model_info': {
                    'model_version': 'Xception-98',
                    'validation_accuracy': 98.0,
                    'tta_enabled': diagnosis_data.get('tta_enabled', True),
                    'uncertainty_metrics': diagnosis_data.get('uncertainty_metrics', {})
                },
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Save to diagnoses collection
            doc_ref = self.db.collection(config.firebase.collection_diagnoses).document(diagnosis_id)
            doc_ref.set(diagnosis_doc)
            
            logger.info(f"Diagnosis saved with ID: {diagnosis_id}")
            return diagnosis_id
            
        except Exception as e:
            print(f"Error saving diagnosis: {e}")
            return None
    
    def save_report_info(self, report_data: Dict) -> str:
        """Save report generation information to Firestore."""
        if not self.is_connected():
            return None
        
        try:
            # Generate unique report ID
            report_id = str(uuid.uuid4())
            
            report_doc = {
                'report_id': report_id,
                'diagnosis_id': report_data.get('diagnosis_id', ''),
                'patient_id': report_data.get('patient_id', ''),
                'report_path': report_data.get('report_path', ''),
                'report_type': 'diagnostic_report',
                'generated_at': datetime.now(),
                'status': 'completed',
                'file_size': report_data.get('file_size', 0),
                'pages': report_data.get('pages', 1)
            }
            
            # Save to reports collection
            doc_ref = self.db.collection(config.firebase.collection_reports).document(report_id)
            doc_ref.set(report_doc)
            
            logger.info(f"Report info saved with ID: {report_id}")
            return report_id
            
        except Exception as e:
            print(f"Error saving report info: {e}")
            return None
    
    def create_clinical_session(self, session_data: Dict) -> str:
        """Create a clinical session document."""
        if not self.is_connected():
            return None
        
        try:
            session_id = str(uuid.uuid4())
            
            session_doc = {
                'session_id': session_id,
                'patient_id': session_data.get('patient_id', ''),
                'diagnoses': session_data.get('diagnoses', []),
                'session_date': datetime.now(),
                'clinician_notes': session_data.get('clinician_notes', ''),
                'follow_up_required': session_data.get('follow_up_required', False),
                'follow_up_date': session_data.get('follow_up_date', None),
                'created_at': datetime.now()
            }
            
            doc_ref = self.db.collection(config.firebase.collection_sessions).document(session_id)
            doc_ref.set(session_doc)
            
            logger.info(f"Clinical session created with ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"Error creating clinical session: {e}")
            return None
    
    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """Retrieve patient's diagnosis history."""
        if not self.is_connected():
            return []
        
        try:
            diagnoses_ref = self.db.collection(config.firebase.collection_diagnoses)
            query = diagnoses_ref.where('patient_id', '==', patient_id).order_by('created_at', direction=firestore.Query.DESCENDING)
            
            history = []
            for doc in query.stream():
                diagnosis_data = doc.to_dict()
                history.append(diagnosis_data)
            
            return history
            
        except Exception as e:
            print(f"Error retrieving patient history: {e}")
            return []
    
    def get_diagnosis_stats(self) -> Dict:
        """Get statistics about diagnoses in the database."""
        if not self.is_connected():
            return {}
        
        try:
            diagnoses_ref = self.db.collection(config.firebase.collection_diagnoses)
            docs = diagnoses_ref.stream()
            
            stats = {
                'total_diagnoses': 0,
                'class_distribution': {},
                'confidence_stats': {
                    'high_confidence': 0,  # >85%
                    'medium_confidence': 0,  # 70-85%
                    'low_confidence': 0     # <70%
                },
                'urgency_levels': {}
            }
            
            for doc in docs:
                data = doc.to_dict()
                stats['total_diagnoses'] += 1
                
                # Class distribution
                predicted_class = data.get('predicted_class', 'unknown')
                stats['class_distribution'][predicted_class] = stats['class_distribution'].get(predicted_class, 0) + 1
                
                # Confidence stats
                confidence = data.get('confidence_score', 0)
                if confidence > 85:
                    stats['confidence_stats']['high_confidence'] += 1
                elif confidence > 70:
                    stats['confidence_stats']['medium_confidence'] += 1
                else:
                    stats['confidence_stats']['low_confidence'] += 1
                
                # Urgency levels
                clinical_assessment = data.get('clinical_assessment', {})
                urgency = clinical_assessment.get('urgency_level', 'unknown')
                stats['urgency_levels'][urgency] = stats['urgency_levels'].get(urgency, 0) + 1
            
            return stats
            
        except Exception as e:
            print(f"Error getting diagnosis stats: {e}")
            return {}


def get_firebase_manager() -> FirebaseManager:
    """Get Firebase manager instance using centralized configuration.
    
    Creates manager with config-based initialization for consistent setup.
    """
    return FirebaseManager()


def save_clinical_data_to_firebase(patient_info: Optional[Dict], 
                                 diagnosis_data: Dict, 
                                 report_path: Optional[str] = None,
                                 firebase_manager: Optional[FirebaseManager] = None) -> Dict:
    """Save clinical data to Firebase Firestore.
    
    Convenience wrapper for saving patient info, diagnosis, report metadata, and clinical
    session to Firebase using centralized configuration. Creates new manager if not provided.
    
    Returns dictionary with saved document IDs or empty dict if Firebase unavailable.
    """
    if firebase_manager is None:
        firebase_manager = get_firebase_manager()
    
    if not firebase_manager.is_connected():
        print("Firebase not connected. Data not saved to cloud.")
        return {}
    
    result = {}
    
    try:
        # Save patient info if provided
        patient_id = None
        if patient_info:
            patient_id = firebase_manager.save_patient_info(patient_info)
            result['patient_id'] = patient_id
        
        # Add patient_id to diagnosis data
        if patient_id:
            diagnosis_data['patient_id'] = patient_id
        
        # Save diagnosis
        diagnosis_id = firebase_manager.save_diagnosis(diagnosis_data)
        result['diagnosis_id'] = diagnosis_id
        
        # Save report info if available
        if report_path and diagnosis_id:
            report_data = {
                'diagnosis_id': diagnosis_id,
                'patient_id': patient_id or 'unknown',
                'report_path': report_path,
                'file_size': os.path.getsize(report_path) if os.path.exists(report_path) else 0
            }
            report_id = firebase_manager.save_report_info(report_data)
            result['report_id'] = report_id
        
        # Create clinical session
        if patient_id and diagnosis_id:
            session_data = {
                'patient_id': patient_id,
                'diagnoses': [diagnosis_id],
                'follow_up_required': 'URGENT' in diagnosis_data.get('clinical_assessment', {}).get('urgency_level', '')
            }
            session_id = firebase_manager.create_clinical_session(session_data)
            result['session_id'] = session_id
        
        logger.info(f"Clinical data saved to Firebase: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error saving clinical data to Firebase: {e}")
        return {}


if __name__ == "__main__":
    # Test Firebase connection and create collections
    print("Testing Firebase integration...")
    
    if firebase_manager.is_connected():
        print("Creating collections structure...")
        firebase_manager.create_collections_structure()
        
        # Test saving sample data
        sample_patient = {
            'name': 'Test Patient',
            'dob': '1990-01-01',
            'gender': 'M',
            'contact_info': {'email': 'test@example.com'}
        }
        
        sample_diagnosis = {
            'image_path': 'test/image.jpg',
            'predicted_class': 'normal',
            'confidence_score': 85.5,
            'clinical_assessment': {
                'clinical_confidence': 'High',
                'urgency_level': 'ROUTINE',
                'recommendations': ['Continue routine monitoring']
            }
        }
        
        result = save_clinical_data_to_firebase(sample_patient, sample_diagnosis)
        print(f"Test data saved: {result}")
        
        # Get stats
        stats = firebase_manager.get_diagnosis_stats()
        print(f"Database stats: {stats}")
    else:
        print("Firebase connection failed. Please check configuration.")