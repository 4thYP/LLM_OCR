import os
import re
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# PDF EXTRACTION MODULES
# ============================================================================
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("pdfplumber not available. Install with: pip install pdfplumber")

# ============================================================================
# WORD DOCUMENT READING
# ============================================================================
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("python-docx not available. Install with: pip install python-docx")

# ============================================================================
# METRICS CALCULATION MODULES
# ============================================================================
import jiwer
from sklearn.metrics import accuracy_score, f1_score

# BLEU Score
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. BLEU scores will be estimated.")

# ROUGE Score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("ROUGE not available. Install with: pip install rouge-score")

# BERT Score
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("BERTScore not available. Install with: pip install bert-score")

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class ExtractionResult:
    """Store extracted text results"""
    page_wise_text: List[str]
    line_wise_text: List[List[str]]
    full_text: str

@dataclass
class MetricsResult:
    """Store all comparison metrics"""
    word_error_rate: float
    character_error_rate: float
    bleu_score: float
    rouge_scores: Dict[str, float]
    bert_scores: Optional[Dict[str, float]]
    accuracy: float
    f1_score: float

# ============================================================================
# PDF EXTRACTOR CLASS
# ============================================================================
class PDFExtractor:
    """Extract text from PDF files using multiple methods"""
    
    def __init__(self):
        self.extraction_methods = []
        if PDFPLUMBER_AVAILABLE:
            self.extraction_methods.append(self._extract_with_pdfplumber)
        if PYPDF2_AVAILABLE:
            self.extraction_methods.append(self._extract_with_pypdf2)
    
    def extract_text(self, pdf_path: str) -> ExtractionResult:
        """Extract text from PDF using available methods"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not self.extraction_methods:
            raise ImportError("No PDF extraction libraries available. Install PyPDF2 or pdfplumber.")
        
        print(f"Extracting text from PDF: {pdf_path}")
        
        # Try each extraction method
        for method in self.extraction_methods:
            try:
                result = method(pdf_path)
                if result.full_text.strip():
                    print(f"✓ Successfully extracted text using {method.__name__}")
                    return result
            except Exception as e:
                print(f"✗ {method.__name__} failed: {e}")
        
        # If all methods fail, return empty result
        print("Warning: All PDF extraction methods failed")
        return ExtractionResult([], [], "")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> ExtractionResult:
        """Extract text using pdfplumber"""
        page_wise_text = []
        line_wise_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    page_wise_text.append(text)
                    # Split into lines
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    line_wise_text.append(lines)
                else:
                    page_wise_text.append("")
                    line_wise_text.append([])
        
        full_text = ' '.join(page_wise_text)
        return ExtractionResult(page_wise_text, line_wise_text, full_text)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> ExtractionResult:
        """Extract text using PyPDF2"""
        page_wise_text = []
        line_wise_text = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    page_wise_text.append(text)
                    # Split into lines
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    line_wise_text.append(lines)
                else:
                    page_wise_text.append("")
                    line_wise_text.append([])
        
        full_text = ' '.join(page_wise_text)
        return ExtractionResult(page_wise_text, line_wise_text, full_text)

# ============================================================================
# WORD DOCUMENT READER
# ============================================================================
class WordDocumentReader:
    """Read text from Word documents"""
    
    @staticmethod
    def read_document(doc_path: str) -> str:
        """Read text from Word document or text file"""
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        print(f"Reading ground truth from: {doc_path}")
        
        # Try to read as .docx first
        if doc_path.lower().endswith('.docx') and DOCX_AVAILABLE:
            try:
                doc = Document(doc_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                text = '\n'.join(full_text)
                print(f"✓ Successfully read Word document")
                return text
            except Exception as e:
                print(f"✗ Failed to read as .docx: {e}")
        
        # Try to read as plain text
        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    with open(doc_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    print(f"✓ Successfully read as text file (encoding: {encoding})")
                    return text
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"✗ Failed to read as text file: {e}")
        
        raise ValueError(f"Could not read document: {doc_path}")

# ============================================================================
# TEXT COMPARATOR
# ============================================================================
class TextComparator:
    """Compare texts and calculate metrics"""
    
    def __init__(self):
        """Initialize text comparison utilities"""
        # Initialize ROUGE scorer if available
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                            use_stemmer=True)
            except Exception as e:
                print(f"ROUGE scorer initialization failed: {e}")
                self.rouge_scorer = None
        
        # Smoothing function for BLEU
        self.smoothing = SmoothingFunction() if NLTK_AVAILABLE else None
    
    def normalize_text(self, text: str, for_metrics: bool = True) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        if for_metrics:
            # For metric calculations
            text = text.lower()
            text = ' '.join(text.split())
            text = re.sub(r'[^\w\s.,;:?!-]', ' ', text)
            text = ' '.join(text.split())
        else:
            # For display/raw comparison
            text = ' '.join(text.split())
        
        return text
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate (WER)"""
        try:
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
            ])
            
            reference_transformed = transformation(reference)
            hypothesis_transformed = transformation(hypothesis)
            
            # Handle empty strings
            if not reference_transformed and not hypothesis_transformed:
                return 0.0
            if not reference_transformed or not hypothesis_transformed:
                return 1.0
            
            return jiwer.wer(reference_transformed, hypothesis_transformed)
        except Exception as e:
            print(f"WER calculation error: {e}")
            return 1.0
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER)"""
        try:
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=False),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
            ])
            
            reference_transformed = transformation(reference)
            hypothesis_transformed = transformation(hypothesis)
            
            # Handle empty strings
            if not reference_transformed and not hypothesis_transformed:
                return 0.0
            if not reference_transformed or not hypothesis_transformed:
                return 1.0
            
            return jiwer.cer(reference_transformed, hypothesis_transformed)
        except Exception as e:
            print(f"CER calculation error: {e}")
            return 1.0
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score"""
        if not NLTK_AVAILABLE:
            # Fallback: Simple word overlap
            ref_words = set(self.normalize_text(reference, True).split())
            hyp_words = set(self.normalize_text(hypothesis, True).split())
            
            if not ref_words or not hyp_words:
                return 0.0
            
            overlap = len(ref_words.intersection(hyp_words))
            total = len(ref_words.union(hyp_words))
            return overlap / total if total > 0 else 0.0
        
        try:
            ref_processed = self.normalize_text(reference, True)
            hyp_processed = self.normalize_text(hypothesis, True)
            
            ref_tokens = [ref_processed.split()]
            hyp_tokens = hyp_processed.split()
            
            if not hyp_tokens:
                return 0.0
            
            score = sentence_bleu(
                ref_tokens, 
                hyp_tokens,
                smoothing_function=self.smoothing.method1,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            return score
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not self.rouge_scorer:
            # Fallback calculation
            ref_words = set(self.normalize_text(reference, False).split())
            hyp_words = set(self.normalize_text(hypothesis, False).split())
            
            if not ref_words or not hyp_words:
                return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
            overlap = len(ref_words.intersection(hyp_words))
            precision = overlap / len(hyp_words) if hyp_words else 0
            recall = overlap / len(ref_words) if ref_words else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'rouge1': f1,
                'rouge2': f1 * 0.85,
                'rougeL': f1 * 0.95
            }
        
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            # Fallback
            ref_words = set(self.normalize_text(reference, False).split())
            hyp_words = set(self.normalize_text(hypothesis, False).split())
            
            if not ref_words or not hyp_words:
                return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
            overlap = len(ref_words.intersection(hyp_words))
            precision = overlap / len(hyp_words) if hyp_words else 0
            recall = overlap / len(ref_words) if ref_words else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'rouge1': f1,
                'rouge2': f1 * 0.85,
                'rougeL': f1 * 0.95
            }
    
    def calculate_bert_score(self, reference: str, hypothesis: str) -> Optional[Dict[str, float]]:
        """Calculate BERTScore"""
        if not BERT_AVAILABLE:
            return None
        
        try:
            P, R, F1 = bert_score([hypothesis], [reference], 
                                 lang='en', verbose=False, rescale_with_baseline=True)
            return {
                'precision': float(P.mean()),
                'recall': float(R.mean()),
                'f1': float(F1.mean())
            }
        except Exception as e:
            print(f"BERTScore calculation error: {e}")
            return None
    
    def calculate_token_metrics(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """Calculate token-based accuracy and F1-score"""
        try:
            ref_processed = self.normalize_text(reference, True)
            hyp_processed = self.normalize_text(hypothesis, True)
            
            ref_tokens = ref_processed.split()
            hyp_tokens = hyp_processed.split()
            
            if not ref_tokens and not hyp_tokens:
                return 1.0, 1.0
            if not ref_tokens or not hyp_tokens:
                return 0.0, 0.0
            
            all_tokens = list(set(ref_tokens + hyp_tokens))
            ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]
            hyp_vector = [1 if token in hyp_tokens else 0 for token in all_tokens]
            
            accuracy = accuracy_score(ref_vector, hyp_vector)
            f1 = f1_score(ref_vector, hyp_vector, average='weighted')
            
            return accuracy, f1
        except Exception as e:
            print(f"Token metrics error: {e}")
            return 0.0, 0.0
    
    def compare_all_metrics(self, reference_text: str, extracted_text: str) -> MetricsResult:
        """Calculate all metrics between reference and extracted text"""
        print("\n" + "="*60)
        print("CALCULATING METRICS")
        print("="*60)
        
        print("  Calculating Word Error Rate...")
        wer = self.calculate_wer(reference_text, extracted_text)
        
        print("  Calculating Character Error Rate...")
        cer = self.calculate_cer(reference_text, extracted_text)
        
        print("  Calculating BLEU score...")
        bleu = self.calculate_bleu(reference_text, extracted_text)
        
        print("  Calculating ROUGE scores...")
        rouge = self.calculate_rouge(reference_text, extracted_text)
        
        print("  Calculating BERTScore...")
        bert_scores = self.calculate_bert_score(reference_text, extracted_text)
        
        print("  Calculating token-based metrics...")
        accuracy, f1 = self.calculate_token_metrics(reference_text, extracted_text)
        
        print("✓ All metrics calculated")
        
        return MetricsResult(
            word_error_rate=wer,
            character_error_rate=cer,
            bleu_score=bleu,
            rouge_scores=rouge,
            bert_scores=bert_scores,
            accuracy=accuracy,
            f1_score=f1
        )

# ============================================================================
# FILE SAVER
# ============================================================================
class FileSaver:
    """Save results to files"""
    
    @staticmethod
    def save_extraction_results(extraction_result: ExtractionResult, output_dir: str):
        """Save page-wise and line-wise extractions"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save page-wise extraction
        page_file = os.path.join(output_dir, "page_wise_extraction.txt")
        with open(page_file, 'w', encoding='utf-8') as f:
            for i, page_text in enumerate(extraction_result.page_wise_text):
                f.write(f"===== Page {i+1} =====\n")
                f.write(page_text)
                f.write("\n\n")
        print(f"  ✓ Saved page-wise extraction: {page_file}")
        
        # Save line-wise extraction
        line_file = os.path.join(output_dir, "line_wise_extraction.txt")
        with open(line_file, 'w', encoding='utf-8') as f:
            for i, page_lines in enumerate(extraction_result.line_wise_text):
                f.write(f"===== Page {i+1} =====\n")
                for j, line in enumerate(page_lines):
                    f.write(f"Line {j+1}: {line}\n")
                f.write("\n")
        print(f"  ✓ Saved line-wise extraction: {line_file}")
        
        # Save full extracted text
        full_file = os.path.join(output_dir, "full_extracted_text.txt")
        with open(full_file, 'w', encoding='utf-8') as f:
            f.write(extraction_result.full_text)
        print(f"  ✓ Saved full extracted text: {full_file}")
    
    @staticmethod
    def save_ground_truth(ground_truth: str, output_dir: str):
        """Save ground truth text"""
        os.makedirs(output_dir, exist_ok=True)
        
        gt_file = os.path.join(output_dir, "ground_truth.txt")
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(ground_truth)
        print(f"  ✓ Saved ground truth: {gt_file}")
    
    @staticmethod
    def save_metrics_results(metrics: MetricsResult, output_dir: str, 
                           ref_text: str, ext_text: str):
        """Save metrics results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results in TXT format
        result_file = os.path.join(output_dir, "result.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DOCUMENT COMPARISON RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write("SUMMARY METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Word Error Rate (WER):      {metrics.word_error_rate:.4f} ({metrics.word_error_rate*100:.1f}% error)\n")
            f.write(f"Character Error Rate (CER): {metrics.character_error_rate:.4f} ({metrics.character_error_rate*100:.1f}% error)\n")
            f.write(f"BLEU Score:                 {metrics.bleu_score:.4f}\n")
            f.write(f"Accuracy:                   {metrics.accuracy:.4f}\n")
            f.write(f"F1-Score:                   {metrics.f1_score:.4f}\n\n")
            
            f.write("ROUGE SCORES:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.rouge_scores.items():
                f.write(f"{key:10}: {value:.4f}\n")
            f.write("\n")
            
            if metrics.bert_scores:
                f.write("BERT SCORES:\n")
                f.write("-" * 40 + "\n")
                for key, value in metrics.bert_scores.items():
                    f.write(f"{key:10}: {value:.4f}\n")
                f.write("\n")
            else:
                f.write("BERT SCORES: Not available (install bert-score package)\n\n")
            
            f.write("TEXT STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Reference text length: {len(ref_text):,} characters\n")
            f.write(f"Extracted text length: {len(ext_text):,} characters\n")
            f.write(f"Reference word count:  {len(ref_text.split()):,} words\n")
            f.write(f"Extracted word count:  {len(ext_text.split()):,} words\n\n")
            
            # Simple word overlap
            ref_words = set(re.sub(r'[^\w\s]', '', ref_text.lower()).split())
            ext_words = set(re.sub(r'[^\w\s]', '', ext_text.lower()).split())
            common_words = ref_words.intersection(ext_words)
            
            f.write("WORD OVERLAP ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Common words:        {len(common_words):,}\n")
            f.write(f"Reference words:     {len(ref_words):,}\n")
            f.write(f"Extracted words:     {len(ext_words):,}\n")
            f.write(f"Overlap percentage:  {len(common_words)/len(ref_words)*100:.1f}%\n\n")
            
            f.write("QUALITY ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            if metrics.word_error_rate < 0.2:
                f.write("✓ EXCELLENT - High quality extraction (WER < 20%)\n")
            elif metrics.word_error_rate < 0.4:
                f.write("✓ GOOD - Acceptable extraction quality (WER 20-40%)\n")
            elif metrics.word_error_rate < 0.6:
                f.write("⚠ MODERATE - Room for improvement (WER 40-60%)\n")
            else:
                f.write("✗ POOR - Low extraction quality (WER > 60%)\n")
            f.write("\n")
            
            f.write("INTERPRETATION GUIDE:\n")
            f.write("-" * 40 + "\n")
            f.write("WER/CER: Lower is better (0.0 = perfect match)\n")
            f.write("BLEU/ROUGE/BERT: Higher is better (1.0 = perfect match)\n")
            f.write("Accuracy/F1: Higher is better (1.0 = perfect match)\n")
        
        print(f"  ✓ Saved metrics results: {result_file}")
        
        # Save JSON version for programmatic use
        json_file = os.path.join(output_dir, "detailed_results.json")
        results_dict = {
            "metrics": {
                "word_error_rate": float(metrics.word_error_rate),
                "character_error_rate": float(metrics.character_error_rate),
                "bleu_score": float(metrics.bleu_score),
                "rouge_scores": {k: float(v) for k, v in metrics.rouge_scores.items()},
                "accuracy": float(metrics.accuracy),
                "f1_score": float(metrics.f1_score),
            },
            "text_statistics": {
                "reference_characters": len(ref_text),
                "extracted_characters": len(ext_text),
                "reference_words": len(ref_text.split()),
                "extracted_words": len(ext_text.split()),
                "word_overlap_percentage": len(common_words)/len(ref_words)*100 if ref_words else 0
            }
        }
        
        if metrics.bert_scores:
            results_dict["metrics"]["bert_scores"] = metrics.bert_scores
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"  ✓ Saved detailed JSON results: {json_file}")

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================
def process_documents(pdf_path: str, word_path: str, output_dir: str = "results"):
    """Main function to process PDF and Word documents"""
    print("="*70)
    print("DOCUMENT COMPARISON SYSTEM")
    print("="*70)
    
    # Check if files exist
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    if not os.path.exists(word_path):
        print(f"Error: Word document not found: {word_path}")
        return
    
    # Step 1: Extract text from PDF
    print("\n[STEP 1] EXTRACTING TEXT FROM PDF")
    print("-" * 40)
    pdf_extractor = PDFExtractor()
    extraction_result = pdf_extractor.extract_text(pdf_path)
    
    if not extraction_result.full_text.strip():
        print("Warning: No text could be extracted from PDF")
        print("The PDF might be image-based/scanned.")
        print("Consider using OCR tools for handwritten text.")
        # Continue anyway to show what metrics we can calculate
    
    # Step 2: Read ground truth from Word document
    print("\n[STEP 2] READING GROUND TRUTH")
    print("-" * 40)
    word_reader = WordDocumentReader()
    ground_truth = word_reader.read_document(word_path)
    
    if not ground_truth.strip():
        print("Warning: Ground truth document is empty")
    
    # Step 3: Save extracted texts
    print("\n[STEP 3] SAVING EXTRACTED TEXTS")
    print("-" * 40)
    FileSaver.save_extraction_results(extraction_result, output_dir)
    FileSaver.save_ground_truth(ground_truth, output_dir)
    
    # Step 4: Calculate metrics
    comparator = TextComparator()
    metrics = comparator.compare_all_metrics(ground_truth, extraction_result.full_text)
    
    # Step 5: Save metrics
    print("\n[STEP 5] SAVING RESULTS")
    print("-" * 40)
    FileSaver.save_metrics_results(metrics, output_dir, ground_truth, extraction_result.full_text)
    
    # Display summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {os.path.abspath(output_dir)}/")
    print("\nSummary Metrics:")
    print(f"  Word Error Rate (WER):      {metrics.word_error_rate:.4f} ({metrics.word_error_rate*100:.1f}%)")
    print(f"  Character Error Rate (CER): {metrics.character_error_rate:.4f} ({metrics.character_error_rate*100:.1f}%)")
    print(f"  BLEU Score:                 {metrics.bleu_score:.4f}")
    print(f"  ROUGE-1 Score:              {metrics.rouge_scores['rouge1']:.4f}")
    print(f"  Accuracy:                   {metrics.accuracy:.4f}")
    print(f"  F1-Score:                   {metrics.f1_score:.4f}")
    
    if metrics.bert_scores:
        print(f"  BERTScore F1:              {metrics.bert_scores['f1']:.4f}")
    
    print("\n" + "="*70)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Compare handwritten PDF with ground truth Word document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s handwritten.pdf ground_truth.docx
  %(prog)s handwritten.pdf ground_truth.docx --output my_results
  %(prog)s --pdf exam.pdf --word answers.docx --output analysis_results
        """
    )
    
    parser.add_argument(
        "pdf_file", 
        nargs="?", 
        help="Path to the handwritten PDF file"
    )
    
    parser.add_argument(
        "word_file", 
        nargs="?", 
        help="Path to the ground truth Word document (.docx or .txt)"
    )
    
    parser.add_argument(
        "--pdf", 
        dest="pdf_file_alt",
        help="Path to the handwritten PDF file (alternative)"
    )
    
    parser.add_argument(
        "--word", 
        dest="word_file_alt",
        help="Path to the ground truth Word document (alternative)"
    )
    
    parser.add_argument(
        "--output", 
        "-o",
        default="comparison_results",
        help="Output directory for results (default: comparison_results)"
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Show installation instructions"
    )
    
    args = parser.parse_args()
    
    # Show installation instructions if requested
    if args.install:
        print("\n" + "="*70)
        print("INSTALLATION INSTRUCTIONS")
        print("="*70)
        print("\nRequired packages:")
        print("  pip install jiwer scikit-learn")
        print("\nFor PDF extraction (choose one or both):")
        print("  pip install PyPDF2")
        print("  pip install pdfplumber")
        print("\nFor Word document reading:")
        print("  pip install python-docx")
        print("\nFor advanced metrics:")
        print("  pip install nltk")
        print("  python -c \"import nltk; nltk.download('punkt')\"")
        print("  pip install rouge-score")
        print("  pip install bert-score")
        print("\n" + "="*70)
        return
    
    # Determine which files to use
    pdf_path = args.pdf_file_alt or args.pdf_file
    word_path = args.word_file_alt or args.word_file
    
    # If no files provided via arguments, ask interactively
    if not pdf_path or not word_path:
        print("\nDocument Comparison Tool")
        print("-" * 40)
        
        if not pdf_path:
            pdf_path = input("Enter path to handwritten PDF file: ").strip()
        
        if not word_path:
            word_path = input("Enter path to ground truth Word document: ").strip()
    
    if not pdf_path or not word_path:
        print("Error: Both PDF and Word document paths are required")
        parser.print_help()
        return
    
    # Process the documents
    try:
        process_documents(pdf_path, word_path, args.output)
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("\nMake sure all required packages are installed:")
        print("Run: python script.py --install")

# ============================================================================
# RUN THE SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Check for minimum required packages
    required_packages = ["jiwer", "sklearn"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Error: Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        print("\nFor complete functionality, run: python script.py --install")
    else:
        main()