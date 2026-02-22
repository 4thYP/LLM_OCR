import os
import sys
import requests
from pdf2image import convert_from_path
import pytesseract


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"


def call_ollama_llama31(page_num: int, ocr_text: str) -> str:
    """
    Send OCR text to llama3.1 (via Ollama) and ask it to return a readable
    transcription while preserving line breaks as much as possible.
    """
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
        "options": {
            "temperature": 0.0
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def extract_pdf_handwriting_with_ocr_then_llama(pdf_path: str, output_txt_path: str) -> None:
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for i, img in enumerate(images, start=1):
            # OCR the page image
            ocr_text = pytesseract.image_to_string(img)

            # Send OCR text to llama3.1 (Ollama) for page-wise readable output
            final_text = call_ollama_llama31(i, ocr_text)

            out.write(f"--- Page {i} ---\n")
            out.write(final_text)
            out.write("\n\n")

    print(f"Done. Output saved to: {output_txt_path}")


def main():
    pdf_path = input("Enter the path to the input PDF file: ").strip().strip('"').strip("'")
    output_txt_path = input("Enter the path for the output .txt file: ").strip().strip('"').strip("'")

    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Ensure output directory exists (if a directory is provided)
    out_dir = os.path.dirname(os.path.abspath(output_txt_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        extract_pdf_handwriting_with_ocr_then_llama(pdf_path, output_txt_path)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Ollama at http://localhost:11434")
        print("Make sure you are running: ollama serve")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"ERROR: Ollama HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()