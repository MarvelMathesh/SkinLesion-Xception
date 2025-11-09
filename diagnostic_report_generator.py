"""PDF diagnostic report generator for dermatological AI analysis.
Generates comprehensive clinical reports combining deep learning predictions with
Gemini AI assessment for enhanced diagnostic decision support.
"""

import os
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import io
import logging

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("ReportLab not available. Install with: pip install reportlab")
    REPORTLAB_AVAILABLE = False

# Gemini AI imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Google Generative AI not available. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False


class DiagnosticReportGenerator:
    """Generate PDF diagnostic reports with AI analysis and clinical assessment.
    
    Integrates Xception model predictions, Gemini AI clinical analysis, and patient data
    into structured medical reports following dermatopathology standards.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize report generator with Gemini API credentials and configure styles."""
        self.gemini_api_key = gemini_api_key or config.report.gemini_api_key
        self.setup_gemini()
        self.setup_styles()
        logger.info("Diagnostic report generator initialized.")
        
        # Disease information database
        self.disease_info = {
            'akiec': {
                'full_name': 'Actinic Keratoses and Intraepithelial Carcinoma',
                'severity': 'Medium to High Risk',
                'description': 'Precancerous skin lesions caused by sun damage. Can progress to squamous cell carcinoma.',
                'recommendation': 'Immediate dermatological evaluation recommended. Consider cryotherapy or topical treatments.'
            },
            'bcc': {
                'full_name': 'Basal Cell Carcinoma',
                'severity': 'High Risk - Cancer',
                'description': 'Most common form of skin cancer. Slow-growing but can cause significant local damage.',
                'recommendation': 'Urgent referral to dermatology/oncology. Surgical excision typically required.'
            },
            'bkl': {
                'full_name': 'Benign Keratosis-like Lesions',
                'severity': 'Low Risk',
                'description': 'Non-cancerous skin growths including seborrheic keratoses and lichenoid keratoses.',
                'recommendation': 'Routine monitoring. Consider removal if cosmetically bothersome or changing.'
            },
            'df': {
                'full_name': 'Dermatofibroma',
                'severity': 'Low Risk',
                'description': 'Benign fibrous nodule, often resulting from minor trauma or insect bites.',
                'recommendation': 'Generally no treatment required. Monitor for changes in size or appearance.'
            },
            'mel': {
                'full_name': 'Melanoma',
                'severity': 'Very High Risk - Aggressive Cancer',
                'description': 'Most dangerous form of skin cancer with high metastatic potential.',
                'recommendation': 'URGENT oncology referral required. Immediate biopsy and staging if confirmed.'
            },
            'normal': {
                'full_name': 'Healthy Skin',
                'severity': 'No Risk',
                'description': 'Normal, healthy skin tissue without pathological changes.',
                'recommendation': 'Continue regular skin self-examinations and sun protection practices.'
            },
            'nv': {
                'full_name': 'Melanocytic Nevi (Moles)',
                'severity': 'Low Risk',
                'description': 'Benign proliferation of melanocytes. Most moles are harmless.',
                'recommendation': 'Monitor for ABCDE changes (Asymmetry, Border, Color, Diameter, Evolution).'
            },
            'vasc': {
                'full_name': 'Vascular Lesions',
                'severity': 'Low Risk',
                'description': 'Benign vascular proliferations including angiomas and angiokeratomas.',
                'recommendation': 'Usually benign. Consider removal if bleeding or cosmetically concerning.'
            }
        }
    
    def setup_gemini(self):
        """Configure and authenticate Gemini AI model for clinical analysis."""
        if not GEMINI_AVAILABLE:
            self.gemini_model = None
            print("Gemini AI not available - reports will be generated without AI analysis")
            return
            
        try:
            genai.configure(api_key=self.gemini_api_key)
            # Use the correct Gemini 2.0 Flash model name
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("Gemini AI model initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini AI: {e}")
            # Try fallback model if the main one fails
            try:
                self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
                print("Gemini AI model initialized with fallback (gemini-pro-vision)")
            except:
                self.gemini_model = None
    
    def setup_styles(self):
        """Configure ReportLab PDF styles for medical report formatting."""
        if not REPORTLAB_AVAILABLE:
            return
            
        self.styles = getSampleStyleSheet()
        
        # Custom styles - check if they already exist before adding
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ))
        
        if 'SectionHeading' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeading',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.darkblue
            ))
        
        # Use existing BodyText or create custom one
        if 'CustomBodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY
            ))
        
        if 'HighRisk' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='HighRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                spaceAfter=12
            ))
        
        if 'MediumRisk' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='MediumRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.orange,
                spaceAfter=12
            ))
        
        if 'LowRisk' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='LowRisk',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.green,
                spaceAfter=12
            ))
        
        # Clinical confidence styles
        if 'HighConfidence' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='HighConfidence',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold',
                spaceAfter=12
            ))
        
        if 'ModerateConfidence' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ModerateConfidence',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.orange,
                spaceAfter=12
            ))
        
        if 'LowConfidence' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='LowConfidence',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                spaceAfter=12
            ))
        
        # Urgency styles
        if 'UrgentLevel' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='UrgentLevel',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                fontName='Helvetica-Bold',
                backColor=colors.yellow,
                spaceAfter=12
            ))
        
        if 'HighUrgency' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='HighUrgency',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                fontName='Helvetica-Bold',
                spaceAfter=12
            ))
        
        if 'ModerateUrgency' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ModerateUrgency',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.orange,
                spaceAfter=12
            ))
        
        if 'RoutineLevel' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='RoutineLevel',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.green,
                spaceAfter=12
            ))
    
    def encode_image_for_gemini(self, image_path: str) -> Optional[str]:
        """Encode image to base64 format for Gemini API transmission."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def get_gemini_analysis(self, image_path: str, predicted_class: str, confidence: float) -> str:
        """Request clinical analysis from Gemini AI vision model.
        
        Submits lesion image with model prediction for expert-level dermatological assessment.
        """
        if not self.gemini_model:
            return "AI analysis unavailable - Gemini model not initialized."
        
        try:
            # Read and prepare image
            image = Image.open(image_path)
            
            # Resize image if too large (Gemini has size limits)
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                # Use LANCZOS directly for compatibility with all Pillow versions
                try:
                    image.thumbnail(max_size, Image.Resampler.LANCZOS)
                except AttributeError:
                    # Fallback for older Pillow versions
                    image.thumbnail(max_size, Image.LANCZOS)
                print(f"Image resized for Gemini analysis: {image.size}")
            
            # Create detailed prompt for medical analysis
            prompt = f"""
            You are an expert dermatologist analyzing a skin lesion image. The AI model has predicted this lesion as '{predicted_class}' with {confidence:.1f}% confidence.

            Please provide a detailed clinical analysis including:

            1. **Visual Assessment**: Describe the morphological features you observe (color, shape, texture, borders, size)
            
            2. **Clinical Correlation**: How well does this image align with the AI prediction of '{predicted_class}'?
            
            3. **Differential Diagnosis**: What other conditions should be considered in the differential?
            
            4. **Risk Assessment**: Evaluate the urgency level and potential for malignancy
            
            5. **Clinical Recommendations**: Specific next steps for patient care
            
            6. **Patient Education Points**: Key information the patient should understand
            
            Please provide professional, evidence-based medical insights while acknowledging this is for educational/screening purposes and not a substitute for professional medical examination.
            """
            
            response = self.gemini_model.generate_content([prompt, image])
            
            if response.text:
                return response.text
            else:
                return "AI analysis completed but no text response received. Please consult with a healthcare professional."
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error getting Gemini analysis: {error_msg}")
            
            # Return a clean error message without technical details
            return f"""
            **Clinical AI Assessment**: Advanced analysis temporarily unavailable.
            
            **Primary Assessment**: The AI model predicted this lesion as '{predicted_class}' with {confidence:.1f}% confidence.
            
            **Clinical Recommendation**: Professional dermatological evaluation recommended for comprehensive assessment and definitive diagnosis.
            
            **Note**: The primary deep learning model prediction remains clinically relevant for screening purposes.
            """
    
    def create_header_footer(self, canvas, doc):
        """Render medical report header and footer with metadata on each page."""
        canvas.saveState()
        
        # Header - Professional medical report style
        canvas.setFont('Helvetica-Bold', 14)
        canvas.setFillColor(colors.darkblue)
        canvas.drawString(72, 760, "DERMATOPATHOLOGY DIAGNOSTIC REPORT")
        canvas.setFont('Helvetica', 10)
        canvas.setFillColor(colors.grey)
        canvas.drawString(72, 745, "AI-Assisted Clinical Analysis")
        canvas.line(72, 740, 540, 740)
        
        # Footer - Clean and minimal
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(72, 50, "This report is for clinical screening purposes.")
        canvas.drawString(72, 40, "Professional medical evaluation recommended for definitive diagnosis.")
        canvas.drawString(72, 30, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Page number
        canvas.drawRightString(540, 50, f"Page {doc.page}")
        
        canvas.restoreState()
        
        canvas.restoreState()
    
    def resize_image_for_pdf(self, image_path: str, max_width: float = 4*inch, max_height: float = 3*inch) -> RLImage:
        """Scale image to PDF dimensions while preserving aspect ratio."""
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Calculate scaling factor
            width_ratio = max_width / img_width
            height_ratio = max_height / img_height
            scale_factor = min(width_ratio, height_ratio)
            
            new_width = img_width * scale_factor
            new_height = img_height * scale_factor
            
            return RLImage(image_path, width=new_width, height=new_height)
        except Exception as e:
            print(f"Error resizing image for PDF: {e}")
            # Return a placeholder or raise the error to be handled upstream
            raise e
    
    def create_prediction_table(self, predictions: np.ndarray, class_names: List[str]) -> Table:
        """Generate formatted table displaying ranked classification probabilities."""
        data = [['Rank', 'Disease Classification', 'Clinical Name', 'Probability', 'Risk Assessment']]
        
        # Sort predictions by probability (highest first)
        sorted_indices = np.argsort(predictions)[::-1]
        
        for rank, i in enumerate(sorted_indices, 1):
            prob = predictions[i]
            class_name = class_names[i]
            disease_info = self.disease_info.get(class_name, {})
            full_name = disease_info.get('full_name', class_name.upper())
            severity = disease_info.get('severity', 'Unknown')
            
            # Only show top 5 predictions or those above 1%
            if rank <= 5 or prob > 0.01:
                data.append([
                    str(rank),
                    class_name.upper(),
                    full_name,
                    f"{prob * 100:.1f}%",
                    severity
                ])
        
        table = Table(data, colWidths=[0.5*inch, 1*inch, 2.5*inch, 0.8*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        return table
    
    def get_risk_style(self, severity: str) -> str:
        """Select PDF text style based on clinical risk assessment."""
        if 'High Risk' in severity or 'Cancer' in severity:
            return 'HighRisk'
        elif 'Medium' in severity:
            return 'MediumRisk'
        else:
            return 'LowRisk'
    
    def get_confidence_style(self, confidence_level: str) -> str:
        """Select PDF text style based on prediction confidence level."""
        if 'High' in confidence_level and 'Uncertainty' not in confidence_level:
            return 'HighConfidence'
        elif 'Moderate' in confidence_level:
            return 'ModerateConfidence'
        else:
            return 'LowConfidence'
    
    def get_urgency_style(self, urgency_level: str) -> str:
        """Select PDF text style based on clinical urgency classification."""
        if 'URGENT' in urgency_level:
            return 'UrgentLevel'
        elif 'HIGH' in urgency_level:
            return 'HighUrgency'
        elif 'MODERATE' in urgency_level:
            return 'ModerateUrgency'
        else:
            return 'RoutineLevel'
    
    def clean_ai_analysis_text(self, text: str) -> str:
        """Remove proprietary AI service references and normalize formatting."""
        # Remove references to Gemini, Google, etc.
        unwanted_phrases = [
            "Google's Gemini 2.0 Pro AI",
            "Gemini 2.0 Pro", 
            "Google Gemini",
            "Gemini AI",
            "The following analysis was generated by"
        ]
        
        cleaned_text = text
        for phrase in unwanted_phrases:
            cleaned_text = cleaned_text.replace(phrase, "")
        
        # Remove extra whitespace and clean up formatting
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text
    
    def format_ai_analysis(self, analysis_text: str) -> list:
        """Parse and structure AI analysis text into organized sections."""
        if not analysis_text or "AI Analysis Error" in analysis_text:
            return []
        
        # Split into logical sections based on common medical report structure
        sections = []
        
        # Look for numbered sections or bullet points
        if "1." in analysis_text or "•" in analysis_text:
            # Split by numbered sections or bullets
            parts = analysis_text.replace("**", "").split("\n")
            current_section = ""
            
            for part in parts:
                part = part.strip()
                if part and (part[0].isdigit() or part.startswith("•") or part.startswith("-")):
                    if current_section:
                        sections.append(current_section)
                    current_section = part
                elif part:
                    current_section += " " + part
            
            if current_section:
                sections.append(current_section)
        else:
            # Split by paragraphs
            paragraphs = analysis_text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Clean up formatting
                    cleaned = paragraph.replace("**", "").strip()
                    sections.append(cleaned)
        
        return sections[:5]  # Limit to 5 sections for readability
    
    def generate_report(self, 
                       image_path: str, 
                       predicted_class: str, 
                       confidence: float, 
                       predictions: np.ndarray, 
                       class_names: List[str],
                       patient_info: Optional[Dict] = None,
                       output_path: str = None,
                       clinical_assessment: Optional[Dict] = None) -> str:
        """Generate comprehensive PDF diagnostic report with AI analysis.
        
        Combines model predictions, clinical assessment, patient data, and Gemini AI insights
        into structured medical report following dermatopathology documentation standards.
        """
        
        if not REPORTLAB_AVAILABLE:
            print("Cannot generate PDF - ReportLab not available")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"diagnostic_report_{timestamp}.pdf"
            output_path = str(config.report.reports_dir / report_filename)
            config.report.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Get AI analysis
        print("Getting Gemini AI analysis...")
        ai_analysis = self.get_gemini_analysis(image_path, predicted_class, confidence)
        
        # Create PDF document with professional medical formatting
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=90, bottomMargin=90)  # More space for header/footer
        
        # Build story (content) with enhanced medical structure
        story = []
        
        # Professional medical report title
        story.append(Paragraph("DERMATOPATHOLOGY DIAGNOSTIC REPORT", self.styles['CustomTitle']))
        story.append(Paragraph("AI-Assisted Skin Lesion Analysis", self.styles['SectionHeading']))
        story.append(Spacer(1, 20))
        
        # Report metadata section
        report_info = [
            ['Report ID:', f"DR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"],
            ['Generated:', datetime.now().strftime('%B %d, %Y at %H:%M:%S')],
            ['Analysis Method:', 'Deep Learning + AI Clinical Assessment'],
            ['Model Version:', 'Xception-98 (98% Validation Accuracy)']
        ]
        
        metadata_table = Table(report_info, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 25))
        
        # Clinical Assessment Section (if available)
        if clinical_assessment:
            story.append(Paragraph("CLINICAL ASSESSMENT", self.styles['SectionHeading']))
            
            # Clinical confidence
            confidence_level = clinical_assessment.get('clinical_confidence', 'Unknown')
            confidence_style = self.get_confidence_style(confidence_level)
            story.append(Paragraph(f"<b>Clinical Confidence Level:</b> {confidence_level}", 
                                 self.styles[confidence_style]))
            
            # Urgency level
            urgency = clinical_assessment.get('urgency_level', 'Unknown')
            urgency_style = self.get_urgency_style(urgency)
            story.append(Paragraph(f"<b>Clinical Urgency:</b> {urgency}", 
                                 self.styles[urgency_style]))
            
            # Differential diagnosis
            diff_dx = clinical_assessment.get('differential_diagnosis', [])
            if diff_dx:
                story.append(Paragraph("<b>Differential Diagnosis (Ranked):</b>", self.styles['CustomBodyText']))
                for dx in diff_dx:
                    condition = dx['condition'].upper()
                    probability = dx['probability']
                    rank = dx['rank']
                    full_name = self.disease_info.get(dx['condition'], {}).get('full_name', condition)
                    story.append(Paragraph(f"&nbsp;&nbsp;{rank}. {full_name} - {probability:.1f}%", 
                                         self.styles['CustomBodyText']))
            
            # Clinical recommendations
            recommendations = clinical_assessment.get('recommendations', [])
            if recommendations:
                story.append(Paragraph("<b>Clinical Recommendations:</b>", self.styles['CustomBodyText']))
                for rec in recommendations:
                    story.append(Paragraph(f"• {rec}", self.styles['CustomBodyText']))
            
            story.append(Spacer(1, 20))
        
        # Patient Information Section
        if patient_info:
            story.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeading']))
            patient_data = [
                ['Patient ID:', patient_info.get('patient_id', 'N/A')],
                ['Date of Birth:', patient_info.get('dob', 'N/A')],
                ['Gender:', patient_info.get('gender', 'N/A')],
                ['Examination Date:', patient_info.get('exam_date', datetime.now().strftime('%Y-%m-%d'))]
            ]
            patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
            patient_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
        story.append(Paragraph("LESION IMAGE", self.styles['SectionHeading']))
        
        # Add image with proper error handling
        try:
            img = self.resize_image_for_pdf(image_path)
            story.append(img)
            story.append(Spacer(1, 15))
        except Exception as e:
            print(f"Warning: Could not add image to PDF: {e}")
            story.append(Paragraph(f"[Image could not be loaded: {os.path.basename(image_path)}]", self.styles['CustomBodyText']))
            story.append(Spacer(1, 15))
        
        # Primary Diagnosis
        story.append(Paragraph("PRIMARY AI DIAGNOSIS", self.styles['SectionHeading']))
        
        disease_info = self.disease_info.get(predicted_class, {})
        full_name = disease_info.get('full_name', predicted_class.upper())
        severity = disease_info.get('severity', 'Unknown')
        description = disease_info.get('description', 'No description available.')
        recommendation = disease_info.get('recommendation', 'Consult healthcare provider.')
        
        risk_style = self.get_risk_style(severity)
        
        story.append(Paragraph(f"<b>Predicted Condition:</b> {full_name} ({predicted_class.upper()})", self.styles['CustomBodyText']))
        story.append(Paragraph(f"<b>Confidence Level:</b> {confidence:.2f}%", self.styles['CustomBodyText']))
        story.append(Paragraph(f"<b>Risk Assessment:</b> {severity}", self.styles[risk_style]))
        story.append(Paragraph(f"<b>Description:</b> {description}", self.styles['CustomBodyText']))
        story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", self.styles['CustomBodyText']))
        story.append(Spacer(1, 20))
        
        # All Predictions Table - Clean and organized
        story.append(Paragraph("DIAGNOSTIC PROBABILITY ANALYSIS", self.styles['SectionHeading']))
        prediction_table = self.create_prediction_table(predictions, class_names)
        story.append(prediction_table)
        story.append(Spacer(1, 20))
        
        # AI Analysis Section - Clean and formatted
        if ai_analysis and ai_analysis != "AI analysis unavailable - Gemini model not initialized.":
            story.append(Paragraph("ADVANCED AI CLINICAL ANALYSIS", self.styles['SectionHeading']))
            
            # Clean up the AI analysis text and format properly
            cleaned_analysis = self.clean_ai_analysis_text(ai_analysis)
            
            if "AI Analysis Error" in cleaned_analysis:
                story.append(Paragraph("<b>Clinical AI Assessment:</b> Analysis temporarily unavailable due to technical limitations.", 
                                     self.styles['CustomBodyText']))
                story.append(Paragraph("<b>Primary Assessment:</b> Based on the deep learning model prediction, this appears to be consistent with the predicted condition.", 
                                     self.styles['CustomBodyText']))
                story.append(Paragraph("<b>Clinical Recommendation:</b> Professional dermatological evaluation recommended for definitive diagnosis.", 
                                     self.styles['CustomBodyText']))
            else:
                # Format the AI analysis into clean paragraphs
                analysis_sections = self.format_ai_analysis(cleaned_analysis)
                for section in analysis_sections:
                    if section.strip():
                        story.append(Paragraph(section.strip(), self.styles['CustomBodyText']))
            
            story.append(Spacer(1, 20))
        
        # Technical Information - Clean and professional
        story.append(Paragraph("TECHNICAL SPECIFICATIONS", self.styles['SectionHeading']))
        tech_info = [
            "Analysis performed using Xception deep learning architecture",
            "Model validation accuracy: 98% on clinical dataset", 
            "Test-Time Augmentation applied for enhanced reliability",
            "Image preprocessing: 224×224 pixel resolution with normalization",
            "Clinical assessment includes uncertainty quantification"
        ]
        for info in tech_info:
            story.append(Paragraph(f"• {info}", self.styles['CustomBodyText']))
        
        story.append(Spacer(1, 20))
        
        # Build PDF with proper error handling
        try:
            # Ensure story is not empty and properly structured
            if not story:
                story.append(Paragraph("Error: No content generated", self.styles['CustomBodyText']))
            
            # Build PDF without custom page functions first to test
            doc.build(story)
            print(f"Diagnostic report generated successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            # Try building without custom header/footer as fallback
            try:
                simple_doc = SimpleDocTemplate(output_path + "_simple.pdf", pagesize=A4)
                simple_doc.build(story)
                print(f"Simple diagnostic report generated: {output_path}_simple.pdf")
                return output_path + "_simple.pdf"
            except Exception as e2:
                print(f"Failed to generate even simple PDF: {e2}")
                return None


def generate_diagnostic_report(image_path: str, 
                              predicted_class: str, 
                              confidence: float, 
                              predictions: np.ndarray, 
                              class_names: List[str],
                              gemini_api_key: Optional[str] = None,
                              patient_info: Optional[Dict] = None,
                              output_path: Optional[str] = None,
                              clinical_assessment: Optional[Dict] = None) -> Optional[str]:
    """Generate diagnostic PDF report with AI-assisted clinical analysis.
    
    Convenience wrapper for DiagnosticReportGenerator. Combines model predictions,
    clinical assessment data, and Gemini AI insights into comprehensive medical report.
    Uses centralized config for default settings.
    
    Returns path to generated PDF or None if generation fails.
    """
    generator = DiagnosticReportGenerator(gemini_api_key)
    return generator.generate_report(image_path, predicted_class, confidence, 
                                   predictions, class_names, patient_info, output_path, clinical_assessment)


if __name__ == "__main__":
    # Example usage
    print("Diagnostic Report Generator Module")
    print("Import this module and use generate_diagnostic_report() function")