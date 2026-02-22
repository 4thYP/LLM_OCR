# compare_text_docx_metrics.py
# Compares a .txt file (hypothesis) vs a .docx file (reference/ground truth)
# and writes WER, CER, BLEU, ROUGE, and BERTScore into a user-named output .txt.

import os
import sys
import re
from typing import List, Tuple

from docx import Document


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_docx(path: str) -> str:
    doc = Document(path)
    # Preserve paragraph breaks
    return "\n".join(p.text for p in doc.paragraphs)


def normalize_text(s: str) -> str:
    # Keep text readable but normalize whitespace for fair scoring
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def tokenize_words(s: str) -> List[str]:
    # Simple word tokenization
    return re.findall(r"\S+", s)


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    # DP edit distance
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
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
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
    # CER over characters (including spaces/newlines can be noisy; we normalize whitespace first)
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_bleu(reference: str, hypothesis: str) -> Tuple[str, float]:
    """
    Returns (bleu_str, bleu_score_float).
    Uses sacrebleu if available.
    """
    try:
        import sacrebleu
    except ImportError:
        raise RuntimeError("Missing dependency: sacrebleu. Install with: python -m pip install sacrebleu")

    # sacrebleu expects list of hypotheses and list-of-list references
    bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
    return str(bleu), float(bleu.score)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Uses rouge-score (Google).
    Returns dict with ROUGE-1/2/L F1, precision, recall.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise RuntimeError("Missing dependency: rouge-score. Install with: python -m pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    out = {}
    for k, v in scores.items():
        out[k] = {
            "precision": v.precision,
            "recall": v.recall,
            "f1": v.fmeasure,
        }
    return out


def compute_bertscore(reference: str, hypothesis: str) -> dict:
    """
    Uses bert-score package.
    Returns dict with precision/recall/f1 means.
    """
    try:
        from bert_score import score as bertscore
    except ImportError:
        raise RuntimeError("Missing dependency: bert-score. Install with: python -m pip install bert-score")

    # bert-score expects lists of strings
    P, R, F1 = bertscore([hypothesis], [reference], lang="en", rescale_with_baseline=True)
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
    }


def main():
    txt_path = input("Enter the path to the input TXT file: ").strip().strip('"').strip("'")
    docx_path = input("Enter the path to the input DOCX file: ").strip().strip('"').strip("'")
    output_path = input("Enter the path/name for the output TXT file: ").strip().strip('"').strip("'")

    if not os.path.isfile(txt_path):
        print(f"ERROR: TXT file not found: {txt_path}")
        sys.exit(1)

    if not os.path.isfile(docx_path):
        print(f"ERROR: DOCX file not found: {docx_path}")
        sys.exit(1)

    if not docx_path.lower().endswith(".docx"):
        print("ERROR: Please provide a .docx Word document for the DOC file input.")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Read
    hyp_raw = read_txt(txt_path)      # hypothesis
    ref_raw = read_docx(docx_path)    # reference

    hyp = normalize_text(hyp_raw)
    ref = normalize_text(ref_raw)

    # Compute core edit-distance metrics
    wer = word_error_rate(ref, hyp)
    cer = char_error_rate(ref, hyp)

    # Compute BLEU, ROUGE, BERTScore
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

    # Write results
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

    print(f"Done. Metrics written to: {output_path}")


if __name__ == "__main__":
    main()