import os
import sys
import requests
from docx import Document

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"


def read_docx_text(docx_path: str) -> str:
    """
    Extracts text from a .docx file preserving paragraph breaks.
    """
    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs]
    # Preserve blank lines between empty paragraphs too
    return "\n".join(paragraphs)


def chunk_lines(text: str, max_lines_per_chunk: int = 40) -> list[str]:
    """
    Splits text into chunks of N lines to simulate page-wise output.
    This does NOT render true pages; it's a simple, consistent chunking.
    """
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines_per_chunk):
        chunk = "\n".join(lines[i:i + max_lines_per_chunk]).strip("\n")
        chunks.append(chunk)
    # If the document is empty, keep one chunk
    return chunks if chunks else [""]


def call_ollama_llama31(page_num: int, chunk_text: str) -> str:
    """
    Sends chunk text to llama3.1 via Ollama and requests an exact,
    readable transcription with line breaks preserved.
    """
    prompt = f"""
You are extracting text from a Word document (ground truth).

Rules:
- Output ONLY the extracted text for Page {page_num}.
- Preserve line breaks and spacing as best as possible.
- Do NOT add headings, summaries, explanations, or comments.
- Do NOT rewrite or paraphrase.
- Do NOT invent missing content.

Here is the text for Page {page_num}:
\"\"\"{chunk_text}\"\"\"

Return the extracted text now:
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
    return (data.get("response") or "").strip("\n")


def extract_docx_via_llama(docx_path: str, output_txt_path: str, lines_per_page: int = 40) -> None:
    raw_text = read_docx_text(docx_path)
    pages = chunk_lines(raw_text, max_lines_per_chunk=lines_per_page)

    with open(output_txt_path, "w", encoding="utf-8") as out:
        for i, page_text in enumerate(pages, start=1):
            llama_text = call_ollama_llama31(i, page_text)

            out.write(f"--- Page {i} ---\n")
            out.write(llama_text)
            out.write("\n\n")

    print(f"Done. Output saved to: {output_txt_path}")


def main():
    docx_path = input("Enter the path to the input .docx file: ").strip().strip('"').strip("'")
    output_txt_path = input("Enter the path for the output .txt file: ").strip().strip('"').strip("'")

    if not os.path.isfile(docx_path):
        print(f"ERROR: DOCX file not found: {docx_path}")
        sys.exit(1)

    if not docx_path.lower().endswith(".docx"):
        print("ERROR: Please provide a .docx file (Word document).")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output_txt_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        extract_docx_via_llama(docx_path, output_txt_path, lines_per_page=40)
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