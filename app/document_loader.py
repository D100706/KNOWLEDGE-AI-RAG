from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_document(name, data):
    records = []

    if name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(data))

        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                records.append({
                    "text": text,
                    "metadata": {
                        "source": name,
                        "page": page_no
                    }
                })

    elif name.lower().endswith(".txt"):
        text = data.decode("utf-8", errors="ignore").strip()

        if text:
            records.append({
                "text": text,
                "metadata": {
                    "source": name,
                    "page": 1
                }
            })

    else:
        raise ValueError("Unsupported file type. Use PDF or TXT.")

    return records


def load_uploaded_file(uploaded_file):
    name = uploaded_file.name
    data = uploaded_file.getvalue()
    return extract_document(name, data)


def load_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return extract_document(path.name, path.read_bytes())