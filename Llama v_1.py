# merged_pdf_extract_and_compare.py
#
# 1) Extract handwritten PDF text page-wise using OCR + llama3.1 (Ollama),
#    saving to a user-provided TXT file.
# 2) Compare that TXT (hypothesis) against a user-provided DOCX (reference/ground truth)
#    using WER, CER, BLEU, ROUGE, BERTScore, and write results to a user-provided TXT file.
#
# NOTE: Ensure you have installed:
#   python -m pip install pdf2image pytesseract requests python-docx sacrebleu rouge-score bert-score
# And system deps for pdf2image/tesseract:
#   macOS: brew install poppler tesseract
# And run Ollama:
#   ollama serve
#   ollama pull llama3.1

import os
import sys
import re
from typing import List, Tuple

import requests
from pdf2image import convert_from_path
import pytesseract
from docx import Document


# -----------------------------
# Ollama (llama3.1) extraction
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"


def call_ollama_llama31(page_num: int, ocr_text: str) -> str:
    prompt = f"""
You are extracting text from a handwritten document.

Goal:
- Return ONLY the extracted text for Page {page_num}.
- Keep it readable and comprehensible.
- Preserve line breaks and paragraph breaks as best as possible.
- Do NOT add commentary, headings, summaries, or explanations.
- Do NOT invent content. If a word is unclear, keep it as-is.

Here is the OCR text for Page {page_num}:
\"\"\"{ocr_text}\"\"\"

Return the final extracted text now:
""".strip()

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def extract_pdf_handwriting_with_ocr_then_llama(pdf_path: str, output_txt_path: str) -> None:
    images = convert_from_path(pdf_path)

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for i, img in enumerate(images, start=1):
            ocr_text = pytesseract.image_to_string(img)
            final_text = call_ollama_llama31(i, ocr_text)

            out.write(f"--- Page {i} ---\n")
            out.write(final_text)
            out.write("\n\n")

    print(f"[1/2] PDF extracted text saved to: {output_txt_path}")


# -----------------------------
# Metrics: WER, CER, BLEU, ROUGE, BERTScore
# -----------------------------
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"\S+", s)


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = cur
    return prev[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(hypothesis)
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_bleu(reference: str, hypothesis: str) -> Tuple[str, float]:
    try:
        import sacrebleu
    except ImportError:
        raise RuntimeError("Missing dependency: sacrebleu. Install with: python -m pip install sacrebleu")

    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
    return str(bleu), float(bleu.score)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise RuntimeError("Missing dependency: rouge-score. Install with: python -m pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    out = {}
    for k, v in scores.items():
        out[k] = {"precision": v.precision, "recall": v.recall, "f1": v.fmeasure}
    return out


def compute_bertscore(reference: str, hypothesis: str) -> dict:
    try:
        from bert_score import score as bertscore
    except ImportError:
        raise RuntimeError("Missing dependency: bert-score. Install with: python -m pip install bert-score")

    P, R, F1 = bertscore([hypothesis], [reference], lang="en", rescale_with_baseline=True)
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
    }


def compare_txt_vs_docx_and_write_metrics(txt_path: str, docx_path: str, output_path: str) -> None:
    hyp_raw = read_txt(txt_path)      # hypothesis
    ref_raw = read_docx(docx_path)    # reference

    hyp = normalize_text(hyp_raw)
    ref = normalize_text(ref_raw)

    wer = word_error_rate(ref, hyp)
    cer = char_error_rate(ref, hyp)

    errors = []
    bleu_str = None
    bleu_score = None
    rouge = None
    bert = None

    try:
        bleu_str, bleu_score = compute_bleu(ref, hyp)
    except Exception as e:
        errors.append(str(e))

    try:
        rouge = compute_rouge(ref, hyp)
    except Exception as e:
        errors.append(str(e))

    try:
        bert = compute_bertscore(ref, hyp)
    except Exception as e:
        errors.append(str(e))

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("Text Comparison Metrics\n")
        out.write("=======================\n\n")
        out.write(f"Reference (DOCX): {docx_path}\n")
        out.write(f"Hypothesis (TXT): {txt_path}\n\n")

        out.write("Edit-distance metrics\n")
        out.write("---------------------\n")
        out.write(f"WER (Word Error Rate): {wer:.6f}\n")
        out.write(f"CER (Character Error Rate): {cer:.6f}\n\n")

        out.write("BLEU\n")
        out.write("----\n")
        if bleu_str is not None:
            out.write(f"{bleu_str}\n")
            out.write(f"BLEU score: {bleu_score:.6f}\n\n")
        else:
            out.write("Not computed.\n\n")

        out.write("ROUGE\n")
        out.write("-----\n")
        if rouge is not None:
            for k in ["rouge1", "rouge2", "rougeL"]:
                v = rouge[k]
                out.write(f"{k}: P={v['precision']:.6f} R={v['recall']:.6f} F1={v['f1']:.6f}\n")
            out.write("\n")
        else:
            out.write("Not computed.\n\n")

        out.write("BERTScore\n")
        out.write("---------\n")
        if bert is not None:
            out.write(f"Precision: {bert['precision']:.6f}\n")
            out.write(f"Recall:    {bert['recall']:.6f}\n")
            out.write(f"F1:        {bert['f1']:.6f}\n\n")
        else:
            out.write("Not computed.\n\n")

        if errors:
            out.write("Notes / Missing dependencies\n")
            out.write("----------------------------\n")
            for err in errors:
                out.write(f"- {err}\n")
            out.write("\n")
            out.write("Install missing packages (if needed):\n")
            out.write("  python -m pip install sacrebleu rouge-score bert-score\n")

    print(f"[2/2] Metrics written to: {output_path}")


# -----------------------------
# Main (merged flow)
# -----------------------------
def main():
    pdf_path = input("Enter the path to the input PDF file: ").strip().strip('"').strip("'")
    extracted_txt_path = input("Enter the path/name for the extracted output TXT file: ").strip().strip('"').strip("'")

    docx_path = input("Enter the path to the ground-truth DOCX file: ").strip().strip('"').strip("'")
    metrics_output_path = input("Enter the path/name for the metrics output TXT file: ").strip().strip('"').strip("'")

    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)

    if not os.path.isfile(docx_path):
        print(f"ERROR: DOCX file not found: {docx_path}")
        sys.exit(1)

    if not docx_path.lower().endswith(".docx"):
        print("ERROR: Please provide a .docx Word document for the ground truth.")
        sys.exit(1)

    # Ensure output dirs exist
    for out_path in [extracted_txt_path, metrics_output_path]:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    # Step 1: extract
    try:
        extract_pdf_handwriting_with_ocr_then_llama(pdf_path, extracted_txt_path)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Ollama at http://localhost:11434")
        print("Make sure you are running: ollama serve")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"ERROR: Ollama HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR (extraction): {e}")
        sys.exit(1)

    # Step 2: compare
    try:
        compare_txt_vs_docx_and_write_metrics(extracted_txt_path, docx_path, metrics_output_path)
    except Exception as e:
        print(f"ERROR (metrics): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()