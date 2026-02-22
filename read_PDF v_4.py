import os
import sys
import base64
import requests
from io import BytesIO

from pdf2image import convert_from_path
import pytesseract


OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3.1"

# OPTIONAL: set to a vision-capable model in Ollama (e.g. "llava" or "llama3.2-vision")
# If empty / None, visual description is skipped.
VISION_MODEL = os.environ.get("VISION_MODEL", "").strip() or None


def _ollama_generate(model: str, prompt: str, images_b64: list[str] | None = None, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    if images_b64:
        payload["images"] = images_b64

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def call_llama_for_clean_text(page_num: int, ocr_text: str) -> str:
    """
    Use llama3.1 to output clean, readable extracted text while preserving line breaks.
    """
    prompt = f"""
You are extracting text from a handwritten document.

Rules:
- Return ONLY the extracted text for Page {page_num}.
- Keep it readable and comprehensible.
- Preserve line breaks and paragraph breaks as best as possible.
- Do NOT add commentary, headings, summaries, or explanations.
- Do NOT invent content. If a word is unclear, keep it as-is.

OCR text for Page {page_num}:
\"\"\"{ocr_text}\"\"\"

Return the final extracted text now:
""".strip()

    return _ollama_generate(TEXT_MODEL, prompt, images_b64=None, timeout=300)


def call_vision_for_visuals(page_num: int, page_png_b64: str) -> str:
    """
    Use a vision model (if available) to describe diagrams/drawings and interpret graphs.
    """
    prompt = f"""
You are analyzing Page {page_num} of a document.

Task:
1) If the page contains a diagram / drawing / image: describe what it is communicating in clear words.
2) If the page contains a graph / chart: explain what it means and how to interpret it (axes, trends, comparisons, key takeaways).
3) If there are labels, legends, captions, or numbers, include them.
4) Be factual. If something is unclear, say it is unclear rather than guessing.
5) Output ONLY the description. No extra commentary.

Now analyze the page image.
""".strip()

    return _ollama_generate(VISION_MODEL, prompt, images_b64=[page_png_b64], timeout=300)


def image_to_png_base64(img) -> str:
    """
    Convert a PIL image to base64-encoded PNG for Ollama vision models.
    """
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_pdf_handwriting_with_optional_visuals(pdf_path: str, output_txt_path: str) -> None:
    images = convert_from_path(pdf_path)

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for i, img in enumerate(images, start=1):
            # OCR the page image
            ocr_text = pytesseract.image_to_string(img)

            # Clean / preserve readable text via llama3.1
            final_text = call_llama_for_clean_text(i, ocr_text)

            # Optional: describe visuals via a vision model
            visuals_text = ""
            if VISION_MODEL:
                page_png_b64 = image_to_png_base64(img)
                try:
                    visuals_text = call_vision_for_visuals(i, page_png_b64)
                except requests.HTTPError as e:
                    visuals_text = f"[Visual description skipped: vision model error: {e}]"
                except Exception as e:
                    visuals_text = f"[Visual description skipped: {e}]"
            else:
                visuals_text = "[Visual description skipped: no vision model configured. Set VISION_MODEL env var to a vision model (e.g. llava).]"

            # Write output page-wise
            out.write(f"--- Page {i} ---\n")

            out.write("[Extracted Text]\n")
            out.write(final_text.strip() + "\n\n")

            out.write("[Diagrams / Graphs / Drawings Description]\n")
            out.write(visuals_text.strip() + "\n\n")

    print(f"Done. Output saved to: {output_txt_path}")


def main():
    pdf_path = input("Enter the path to the input PDF file: ").strip().strip('"').strip("'")
    output_txt_path = input("Enter the path for the output .txt file: ").strip().strip('"').strip("'")

    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output_txt_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        extract_pdf_handwriting_with_optional_visuals(pdf_path, output_txt_path)
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