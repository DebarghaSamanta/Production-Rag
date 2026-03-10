from pathlib import Path
import fitz
import markdown
import re


def load_pdf(path: Path) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def load_markdown(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    html = markdown.markdown(raw)
    return re.sub(r"<[^>]+>", "", html)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


LOADERS = {
    ".pdf": load_pdf,
    ".md":  load_markdown,
    ".txt": load_text,
}


def load_document(path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file type: {suffix}")

    text = LOADERS[suffix](path)

    return {
        "text":      text,
        "source":    path.name,
        "file_path": str(path),
        "file_type": suffix,
    }


def load_directory(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    docs = []
    for path in dir_path.iterdir():
        if path.suffix.lower() in LOADERS:
            docs.append(load_document(path))
    return docs