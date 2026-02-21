import google.generativeai as genai
import os
import tempfile
from PIL import Image
import io
import time
from pdf2image import convert_from_path
from docx import Document
import re
from difflib import SequenceMatcher
import json

class HandwritingAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
    def pdf_to_images(self, pdf_path, dpi=200):
        """Convert PDF pages to images"""
        print(f"Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} pages")
        return images
    
    def extract_text_from_image(self, image, max_retries=3):
        """Extract text from image using Gemini"""
        for attempt in range(max_retries):
            try:
                prompt = """
                Extract ALL text from this handwritten document exactly as written. 
                Preserve:
                - Line breaks and paragraph structure
                - Spelling errors and corrections  
                - Punctuation and capitalization
                - Any symbols or special characters
                
                Return only the extracted text, no additional commentary.
                """
                
                response = self.model.generate_content([prompt, image])
                return response.text.strip()
                
            except Exception as e:
                if "quota" in str(e).lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"Quota exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error: {e}")
                    return ""
        return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from all pages of PDF"""
        images = self.pdf_to_images(pdf_path)
        extracted_texts = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            text = self.extract_text_from_image(image)
            extracted_texts.append(text)
            print(f"Page {i+1} extracted: {len(text)} characters")
            
        return "\n\n".join(extracted_texts)
    
    def read_docx(self, docx_path):
        """Extract text from DOCX file"""
        doc = Document(docx_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
            
        return "\n".join(full_text)
    
    def preprocess_text(self, text):
        """Clean and normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = text.strip()
        
        return text
    
    def calculate_accuracy(self, extracted_text, ground_truth_text):
        """Calculate various accuracy metrics"""
        
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Character-level accuracy
        char_matcher = SequenceMatcher(None, extracted_clean, ground_truth_clean)
        char_similarity = char_matcher.ratio()
        
        # Word-level accuracy
        extracted_words = extracted_clean.split()
        ground_truth_words = ground_truth_clean.split()
        
        word_matcher = SequenceMatcher(None, extracted_words, ground_truth_words)
        word_similarity = word_matcher.ratio()
        
        # Calculate precision, recall, F1 for words
        common_words = set(extracted_words) & set(ground_truth_words)
        
        if len(extracted_words) > 0:
            precision = len(common_words) / len(extracted_words)
        else:
            precision = 0
            
        if len(ground_truth_words) > 0:
            recall = len(common_words) / len(ground_truth_words)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        return {
            'character_accuracy': char_similarity * 100,
            'word_accuracy': word_similarity * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100,
            'extracted_word_count': len(extracted_words),
            'ground_truth_word_count': len(ground_truth_words),
            'common_word_count': len(common_words)
        }
    
    def detailed_comparison(self, extracted_text, ground_truth_text):
        """Provide detailed comparison with differences"""
        
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        extracted_lines = extracted_clean.split('\n')
        ground_truth_lines = ground_truth_clean.split('\n')
        
        differences = []
        
        # Compare line by line
        for i, (ext_line, gt_line) in enumerate(zip(extracted_lines, ground_truth_lines)):
            if ext_line != gt_line:
                differences.append({
                    'line_number': i + 1,
                    'extracted': ext_line,
                    'ground_truth': gt_line,
                    'similarity': SequenceMatcher(None, ext_line, gt_line).ratio() * 100
                })
        
        return differences
    
    def generate_report(self, extracted_text, ground_truth_text, accuracy_metrics, differences):
        """Generate a comprehensive report"""
        
        report = {
            'summary': {
                'character_accuracy': f"{accuracy_metrics['character_accuracy']:.2f}%",
                'word_accuracy': f"{accuracy_metrics['word_accuracy']:.2f}%",
                'f1_score': f"{accuracy_metrics['f1_score']:.2f}%",
                'extracted_words': accuracy_metrics['extracted_word_count'],
                'ground_truth_words': accuracy_metrics['ground_truth_word_count']
            },
            'accuracy_metrics': accuracy_metrics,
            'sample_comparison': {
                'first_100_chars_extracted': extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text,
                'first_100_chars_ground_truth': ground_truth_text[:100] + "..." if len(ground_truth_text) > 100 else ground_truth_text
            },
            'major_differences': differences[:5]  # Show first 5 differences
        }
        
        return report

def main():
    # Configuration
    API_KEY = "AIzaSyC0qdImg8YBPG9e4a8uWthx5aS7Q-V8kwo"  # Replace with your API key
    
    # File paths
    handwritten_pdf_path = "handwritten.pdf"  # Replace with your PDF path
    ground_truth_docx_path = "ground_truth.docx"  # Replace with your DOCX path
    
    # Initialize analyzer
    analyzer = HandwritingAnalyzer(API_KEY)
    
    try:
        print("🚀 Starting handwriting analysis...")
        
        # Step 1: Extract text from handwritten PDF
        print("\n📄 Extracting text from handwritten PDF...")
        extracted_text = analyzer.extract_text_from_pdf(handwritten_pdf_path)
        
        print(f"✅ Extracted {len(extracted_text)} characters from PDF")
        
        # Step 2: Read ground truth from DOCX
        print("\n📝 Reading ground truth from DOCX...")
        ground_truth_text = analyzer.read_docx(ground_truth_docx_path)
        
        print(f"✅ Ground truth has {len(ground_truth_text)} characters")
        
        # Step 3: Calculate accuracy metrics
        print("\n📊 Calculating accuracy metrics...")
        accuracy_metrics = analyzer.calculate_accuracy(extracted_text, ground_truth_text)
        
        # Step 4: Detailed comparison
        print("\n🔍 Performing detailed comparison...")
        differences = analyzer.detailed_comparison(extracted_text, ground_truth_text)
        
        # Step 5: Generate report
        print("\n📈 Generating report...")
        report = analyzer.generate_report(extracted_text, ground_truth_text, accuracy_metrics, differences)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 ACCURACY RESULTS")
        print("="*60)
        
        print(f"\n📊 SUMMARY:")
        print(f"Character Accuracy: {report['summary']['character_accuracy']}")
        print(f"Word Accuracy: {report['summary']['word_accuracy']}")
        print(f"F1 Score: {report['summary']['f1_score']}")
        print(f"Extracted Words: {report['summary']['extracted_words']}")
        print(f"Ground Truth Words: {report['summary']['ground_truth_words']}")
        
        print(f"\n📝 SAMPLE COMPARISON:")
        print(f"Extracted: {report['sample_comparison']['first_100_chars_extracted']}")
        print(f"Ground Truth: {report['sample_comparison']['first_100_chars_ground_truth']}")
        
        if differences:
            print(f"\n❌ MAJOR DIFFERENCES (showing {len(report['major_differences'])} of {len(differences)}):")
            for diff in report['major_differences']:
                print(f"Line {diff['line_number']}:")
                print(f"  Extracted: {diff['extracted']}")
                print(f"  Expected:  {diff['ground_truth']}")
                print(f"  Similarity: {diff['similarity']:.1f}%")
                print()
        
        # Save results to file
        output_file = "handwriting_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Full results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
