import os
import re
import json
import string
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# Text comparison metrics
import jiwer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from sklearn.metrics import accuracy_score, f1_score

# For BERT Score (requires torch)
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("BERTScore not available. Install with: pip install bert-score")

# For ROUGE Score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("ROUGE not available. Install with: pip install rouge-score")

@dataclass
class MetricsResult:
    """Store all comparison metrics"""
    word_error_rate: float
    character_error_rate: float
    bleu_score: float
    rouge_scores: Dict[str, float]
    bert_scores: Optional[Tuple[List[float], List[float], List[float]]]
    accuracy: float
    f1_score: float

class TextComparator:
    def __init__(self):
        """Initialize text comparison utilities"""
        # Download necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Initialize ROUGE scorer if available
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.smoothing = SmoothingFunction()
    
    def preprocess_text(self, text: str, normalize: bool = True) -> str:
        """Clean and normalize text for comparison"""
        if not text:
            return ""
        
        if normalize:
            # Convert to lowercase
            text = text.lower()
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,;:?!-]', ' ', text)
            # Remove extra spaces
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
        try:
            # Preprocess texts
            ref_processed = self.preprocess_text(reference, normalize=True)
            hyp_processed = self.preprocess_text(hypothesis, normalize=True)
            
            # Tokenize
            ref_tokens = [ref_processed.split()]
            hyp_tokens = hyp_processed.split()
            
            if not hyp_tokens:
                return 0.0
            
            # Calculate BLEU score with smoothing
            score = sentence_bleu(
                ref_tokens, 
                hyp_tokens,
                smoothing_function=self.smoothing.method1,
                weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4 weights
            )
            return score
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            if not self.rouge_scorer:
                return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
            # Don't normalize for ROUGE to preserve exact matches
            ref_processed = self.preprocess_text(reference, normalize=False)
            hyp_processed = self.preprocess_text(hypothesis, normalize=False)
            
            scores = self.rouge_scorer.score(ref_processed, hyp_processed)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bert_score(self, reference: str, hypothesis: str) -> Optional[Tuple]:
        """Calculate BERTScore"""
        try:
            if not BERT_AVAILABLE:
                return None
            
            # Don't normalize for BERTScore
            ref_processed = self.preprocess_text(reference, normalize=False)
            hyp_processed = self.preprocess_text(hypothesis, normalize=False)
            
            P, R, F1 = bert_score([hyp_processed], [ref_processed], 
                                 lang='en', verbose=False)
            return (P.numpy(), R.numpy(), F1.numpy())
        except Exception as e:
            print(f"BERTScore calculation error: {e}")
            return None
    
    def calculate_token_metrics(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """Calculate token-based accuracy and F1-score"""
        try:
            # Preprocess for token matching
            ref_processed = self.preprocess_text(reference, normalize=True)
            hyp_processed = self.preprocess_text(hypothesis, normalize=True)
            
            # Tokenize
            ref_tokens = ref_processed.split()
            hyp_tokens = hyp_processed.split()
            
            # Handle empty cases
            if not ref_tokens and not hyp_tokens:
                return 1.0, 1.0  # Both empty
            if not ref_tokens or not hyp_tokens:
                return 0.0, 0.0  # One is empty
            
            # Get all unique tokens
            all_tokens = list(set(ref_tokens + hyp_tokens))
            
            # Create binary vectors for token presence
            ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]
            hyp_vector = [1 if token in hyp_tokens else 0 for token in all_tokens]
            
            # Calculate metrics
            accuracy = accuracy_score(ref_vector, hyp_vector)
            f1 = f1_score(ref_vector, hyp_vector, average='weighted')
            
            return accuracy, f1
        except Exception as e:
            print(f"Token metrics error: {e}")
            return 0.0, 0.0
    
    def compare_all_metrics(self, reference_text: str, extracted_text: str) -> MetricsResult:
        """Calculate all metrics between reference and extracted text"""
        
        print("Calculating metrics...")
        
        # Calculate WER and CER
        print("  - Calculating Word Error Rate...")
        wer = self.calculate_wer(reference_text, extracted_text)
        
        print("  - Calculating Character Error Rate...")
        cer = self.calculate_cer(reference_text, extracted_text)
        
        # Calculate BLEU
        print("  - Calculating BLEU score...")
        bleu = self.calculate_bleu(reference_text, extracted_text)
        
        # Calculate ROUGE
        print("  - Calculating ROUGE scores...")
        rouge = self.calculate_rouge(reference_text, extracted_text)
        
        # Calculate BERTScore
        print("  - Calculating BERTScore...")
        bert_scores = self.calculate_bert_score(reference_text, extracted_text)
        
        # Calculate token-based metrics
        print("  - Calculating token-based metrics...")
        accuracy, f1 = self.calculate_token_metrics(reference_text, extracted_text)
        
        return MetricsResult(
            word_error_rate=wer,
            character_error_rate=cer,
            bleu_score=bleu,
            rouge_scores=rouge,
            bert_scores=bert_scores,
            accuracy=accuracy,
            f1_score=f1
        )

def extract_text_from_pdf_simulation(pdf_content: str) -> Tuple[List[str], List[List[str]]]:
    """
    Simulate PDF text extraction based on provided content
    Returns: (page_wise_text, line_wise_text)
    """
    # Parse the provided PDF content
    pages = []
    current_page = []
    
    lines = pdf_content.split('\n')
    for line in lines:
        if line.strip().startswith('===== Page'):
            if current_page:
                pages.append('\n'.join(current_page))
                current_page = []
        else:
            current_page.append(line)
    
    if current_page:
        pages.append('\n'.join(current_page))
    
    # Create line-wise representation
    line_wise = []
    for page in pages:
        page_lines = [line.strip() for line in page.split('\n') if line.strip()]
        line_wise.append(page_lines)
    
    return pages, line_wise

def save_extraction_results(pages: List[str], line_wise: List[List[str]], output_dir: str):
    """Save page-wise and line-wise extractions to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save page-wise extraction
    with open(os.path.join(output_dir, "page_wise_extraction.txt"), 'w', encoding='utf-8') as f:
        for i, page in enumerate(pages):
            f.write(f"===== Page {i+1} =====\n")
            f.write(page)
            f.write("\n\n")
    
    # Save line-wise extraction
    with open(os.path.join(output_dir, "line_wise_extraction.txt"), 'w', encoding='utf-8') as f:
        for i, page_lines in enumerate(line_wise):
            f.write(f"===== Page {i+1} =====\n")
            for j, line in enumerate(page_lines):
                f.write(f"Line {j+1}: {line}\n")
            f.write("\n")
    
    print(f"Saved extraction results to {output_dir}/")

def save_metrics_results(metrics: MetricsResult, output_dir: str, reference_text: str, extracted_text: str):
    """Save metrics results to file"""
    result_content = []
    
    result_content.append("=== METRICS RESULTS ===\n\n")
    
    # Basic metrics
    result_content.append(f"Word Error Rate (WER): {metrics.word_error_rate:.4f} ({metrics.word_error_rate*100:.2f}%)\n")
    result_content.append(f"Character Error Rate (CER): {metrics.character_error_rate:.4f} ({metrics.character_error_rate*100:.2f}%)\n")
    result_content.append(f"BLEU Score: {metrics.bleu_score:.4f}\n")
    result_content.append(f"Accuracy: {metrics.accuracy:.4f}\n")
    result_content.append(f"F1-Score: {metrics.f1_score:.4f}\n\n")
    
    # ROUGE scores
    result_content.append("ROUGE Scores:\n")
    for key, value in metrics.rouge_scores.items():
        result_content.append(f"  {key}: {value:.4f}\n")
    
    result_content.append("\n")
    
    # BERT scores
    if metrics.bert_scores:
        P, R, F1 = metrics.bert_scores
        result_content.append("BERT Scores (averages):\n")
        result_content.append(f"  Precision: {np.mean(P):.4f}\n")
        result_content.append(f"  Recall: {np.mean(R):.4f}\n")
        result_content.append(f"  F1: {np.mean(F1):.4f}\n")
    else:
        result_content.append("BERT Scores: Not available (install bert-score package)\n")
    
    # Text statistics
    result_content.append("\n=== TEXT STATISTICS ===\n")
    result_content.append(f"Reference text length: {len(reference_text)} characters\n")
    result_content.append(f"Extracted text length: {len(extracted_text)} characters\n")
    result_content.append(f"Reference word count: {len(reference_text.split())}\n")
    result_content.append(f"Extracted word count: {len(extracted_text.split())}\n")
    
    # Find differences
    result_content.append("\n=== MAJOR DIFFERENCES ===\n")
    # Simple difference detection
    ref_words = set(reference_text.lower().split())
    ext_words = set(extracted_text.lower().split())
    missing_in_ext = ref_words - ext_words
    extra_in_ext = ext_words - ref_words
    
    result_content.append(f"Words in reference but missing in extraction: {len(missing_in_ext)}\n")
    if len(missing_in_ext) > 0:
        result_content.append(f"  Sample: {', '.join(list(missing_in_ext)[:10])}\n")
    
    result_content.append(f"Extra words in extraction not in reference: {len(extra_in_ext)}\n")
    if len(extra_in_ext) > 0:
        result_content.append(f"  Sample: {', '.join(list(extra_in_ext)[:10])}\n")
    
    # Interpretation
    result_content.append("\n=== INTERPRETATION ===\n")
    result_content.append("WER/CER: Lower is better (0.0 = perfect)\n")
    result_content.append("BLEU/ROUGE/BERT: Higher is better (1.0 = perfect)\n")
    result_content.append("Accuracy/F1: Higher is better (1.0 = perfect)\n")
    
    # Quality assessment
    result_content.append("\n=== QUALITY ASSESSMENT ===\n")
    if metrics.word_error_rate < 0.2:
        result_content.append("Excellent extraction quality (WER < 20%)\n")
    elif metrics.word_error_rate < 0.4:
        result_content.append("Good extraction quality (WER 20-40%)\n")
    elif metrics.word_error_rate < 0.6:
        result_content.append("Moderate extraction quality (WER 40-60%)\n")
    else:
        result_content.append("Poor extraction quality (WER > 60%)\n")
    
    # Save to file
    with open(os.path.join(output_dir, "result.txt"), 'w', encoding='utf-8') as f:
        f.write(''.join(result_content))
    
    # Also save JSON for programmatic use
    results_dict = {
        "word_error_rate": float(metrics.word_error_rate),
        "character_error_rate": float(metrics.character_error_rate),
        "bleu_score": float(metrics.bleu_score),
        "rouge_scores": {k: float(v) for k, v in metrics.rouge_scores.items()},
        "accuracy": float(metrics.accuracy),
        "f1_score": float(metrics.f1_score),
        "text_statistics": {
            "reference_length": len(reference_text),
            "extracted_length": len(extracted_text),
            "reference_words": len(reference_text.split()),
            "extracted_words": len(extracted_text.split())
        }
    }
    
    if metrics.bert_scores:
        P, R, F1 = metrics.bert_scores
        results_dict["bert_scores"] = {
            "precision": float(np.mean(P)),
            "recall": float(np.mean(R)),
            "f1": float(np.mean(F1))
        }
    
    with open(os.path.join(output_dir, "detailed_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Saved metrics results to {output_dir}/")

def main():
    """Main analysis function using your provided documents"""
    
    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # The extracted text from handwritten PDF (based on your provided content)
    pdf_content = """===== Page 1 =====
# TECHNO MAIN SALT LAKE  
( FORMERLY TECHNO INDIA, SALT LAKE )  
**Name:**  
Swanamaki  
Pagamanik  
**Roll No.**  
1.30.30.82.3.13.6  
**Stream:** CS FAIML  
**Subject:**  
Machine Learning Application  
(PCC-ATML 60)  
**Semester:** 6th  
**Invigilator's Signature:**  
Date: 24.03.12.5  
---
## Part A
1. The two most common supervised tasks are regression and classification.
2. The purpose of a validation set is used after training the model. It is used to evaluate the model performance and tuning the model.
3. 2 model parameters are there in a linear regression problem with a single feature variable.
4. The auc value of a perfect classifier is **gp=100%** 1.
5. Precision is more important for a spam email detection system.
---
## Part B
### Train-test-split:
The dataset is splitted into training testing and validation. ♦
The model trained on training set, then the model is tested on a unseen test dataset.

===== Page 2 =====
Overfitting:
When a model learns noise from the training set and model performed well in training set but it does not perform well on unseen data. This indicates overfitting.
Underfitting:
When a model is unable to learn from the training set and the model performs does not well on training set. This called under-fitting.
Prevent them:
a) use a good model.
b) Reduce the noise data.
c) scale the data.
d) Cross-validation.
Confusion matrix:
Confusion matrix is the insight of model performance. It captures includes connectivity and incorrectly predicted data value.
It is important:
as a) It can check accurately from matrix.

===== Page 3 =====
b) It can calculate, Entropy.
c) It also can calculate precision, recall, f-score.
[ th = 82 ]
[ precision = TP/(TP+FP) ]
[ fp = 3 ]
[ fn = 5 ]
[ fp = 10 ]
[ recall = TP/(TP+FN) = 10/(10 + 5) = 10/15 = 0.666 ]
false negative rate = [ FPN/(TP+FN) ] = [ 5/15 ]
false positive rate = [ FPN/(TP+FN) ] = 1 - Precision
= 1 - 0.769 = 0.23
In machine learning model,
Bias is a training terminology in machine learning. It signifies a training model performing poorly in training phase.
Variance is where the model mostly give some errors in testing phase.

===== Page 4 =====
# 11 reduces -
a) to reduce, it take data that
it is properly scaled and the size
of data is measured.
b) to reduce variance, if can
use dimensionality reduction.
---
Bias-variance tradeoff is a scenario
where the model performs poorly
in the training phase."""
    
    # The ground truth text (based on your provided content)
    ground_truth = """Name: Swarnali Pramanik
Roll No.: 13030823136
**Part A**
1.  The two most common supervised tasks are Regression and
    Classification.
2.  The purpose of a validation set is used after training the model. It
    is used to evaluate the model performance and tuning the model.
3.  2 model parameters are there in a linear regression problem with a
    single feature variable.
4.  The AUC value of a perfect classifier is 1.
5.  Precision is more important for a spam email detection system.
**Part B**
6.  The dataset is splitted into training, testing and validation.
The model trained on training set, then the model is tested on unseen
test dataset.
Overfitting:
When a model learn noise data from the training set and the model
performed well on the training set but it does not perform well on
unseen data. This indicates overfitting.
Underfitting:
When a model is unable to learn from the training set and the model
perform does not well on training set. This is called underfitting.
Prevent them :
1.  Use of good model
2.  Reduce the noise from data
3.  Scale the data
4.  Cross-validation
9.  Confusion Matrix:
Confusion matrix is the insight of model performance. It includes
correctly and incorrectly predicted data value.
It is important:
a.  It can check accuracy from matrix.
b.  It can calculate Entropy.
c.  It also can calculate precision, recall and F-score.
d.  True Negative is equal to 82,
False Positive is equal to 3,
False Negative is equal to 5
True Positive is equal to 10
Precision is True Positive, divided by bracket open, True Positive
plus False Positive, bracket close.
Which is equal to, 10 divided by bracket open, 10 plus 3, bracket
close.
Which is equal to, 10 divided 13, bracket close.
Which equals to 0.769, which is the final answer.
Recall is True Positive, divided by bracket open, True Positive plus
False Negative, bracket close.
Which is equal to, 10 divided by bracket open, 10 plus 5, bracket
close.
Which is equal to, 10 divided 13, bracket close.
Which equals to 0.666, which is the final answer.
False Negative Rate is False Negative divided by bracket open, True
Positive plus False Negative, bracket close.
Which is equal to, 5 divided by bracket open, 10 plus 5, bracket
close.
Which is equal to, 5 divided 15, bracket close.
Which equals to 0.333, which is the final answer.
False Positive Rate is 1 minus Precision.
Which is equal to, 1 minus 0.769.
Which equals to 0.231, which is the final answer.
7.  In machine learning model,
Bias is a terminology in machine learning. It signifies a training
model performing poorly in training phase.
Variance is where the model mostly give some error in testing phase.
It reduces:
a.  To reduce, it take data that it is properly scalled and the size of
    the data is measured.
b.  To reduce variance, it can use dimensionally reduction.
Bias-Variance tradeoff is a scenario where the model perform poorly
in the training phase."""
    
    print("="*70)
    print("DOCUMENT COMPARISON AND METRICS CALCULATION")
    print("="*70)
    
    print("\nStep 1: Extracting text from PDF simulation...")
    pages, line_wise = extract_text_from_pdf_simulation(pdf_content)
    print(f"  Extracted {len(pages)} pages")
    
    print("\nStep 2: Saving extraction results...")
    save_extraction_results(pages, line_wise, output_dir)
    
    print("\nStep 3: Preparing texts for comparison...")
    # Combine all pages for overall comparison
    extracted_combined = ' '.join(pages)
    
    # Preprocess ground truth to remove markdown formatting
    ground_truth_clean = re.sub(r'\*\*|__', '', ground_truth)  # Remove bold markers
    ground_truth_clean = ' '.join(ground_truth_clean.split())  # Normalize whitespace
    
    print(f"  Reference text: {len(ground_truth_clean)} characters")
    print(f"  Extracted text: {len(extracted_combined)} characters")
    
    print("\nStep 4: Calculating metrics...")
    comparator = TextComparator()
    metrics = comparator.compare_all_metrics(ground_truth_clean, extracted_combined)
    
    print("\nStep 5: Saving results...")
    save_metrics_results(metrics, output_dir, ground_truth_clean, extracted_combined)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)
    print(f"Word Error Rate (WER):      {metrics.word_error_rate:.4f} ({metrics.word_error_rate*100:.2f}%)")
    print(f"Character Error Rate (CER): {metrics.character_error_rate:.4f} ({metrics.character_error_rate*100:.2f}%)")
    print(f"BLEU Score:                 {metrics.bleu_score:.4f}")
    print(f"Accuracy:                   {metrics.accuracy:.4f}")
    print(f"F1-Score:                   {metrics.f1_score:.4f}")
    
    print("\nROUGE Scores:")
    for key, value in metrics.rouge_scores.items():
        print(f"  {key}: {value:.4f}")
    
    if metrics.bert_scores:
        P, R, F1 = metrics.bert_scores
        print(f"\nBERT Score (average F1): {np.mean(F1):.4f}")
    
    print("\n" + "="*70)
    print(f"All files saved in '{output_dir}' directory:")
    print(f"  ✓ page_wise_extraction.txt")
    print(f"  ✓ line_wise_extraction.txt")
    print(f"  ✓ result.txt")
    print(f"  ✓ detailed_results.json")
    print("="*70)

def install_requirements():
    """Print installation instructions"""
    print("To run this analysis, install the required packages:")
    print("\npip install:")
    print("  jiwer              # For WER and CER")
    print("  nltk               # For BLEU score")
    print("  scikit-learn       # For accuracy and F1-score")
    print("  rouge-score        # For ROUGE metrics (optional)")
    print("  bert-score         # For BERTScore (optional, requires torch)")
    print("\nAdditional setup:")
    print("  import nltk")
    print("  nltk.download('punkt')")

if __name__ == "__main__":
    print("Running document comparison analysis...")
    print("="*70)
    
    # Check if we should install requirements
    try:
        import jiwer
        import nltk
        from sklearn.metrics import accuracy_score, f1_score
        print("✓ Required packages are installed")
        
        # Run the analysis
        main()
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nInstallation required:")
        install_requirements()