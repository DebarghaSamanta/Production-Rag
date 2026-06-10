import os
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import RAW_DATA_DIR


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Replaces the old pypdf extractor with pymupdf4llm.
    
    Why: pypdf cannot handle mathematical notation — superscripts, Greek letters,
    and operators like a^{bc} come out as garbled unicode. pymupdf4llm uses MuPDF's
    layout engine and outputs clean Markdown, which preserves:
      - Exponents as a^bc or a²
      - Greek letters as their actual unicode (α, β, θ)
      - Inline math structure (won't be LaTeX, but won't be garbage either)
      - Multi-column layout order (reads left column first, then right)
      - Section headings as Markdown ## headings
    
    The output is a single Markdown string for the entire PDF.
    """
    print(f"  [Extractor] Parsing with pymupdf4llm: {os.path.basename(pdf_path)}")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text


def split_into_sections(text: str) -> list[str]:
    
    # Primary: Markdown headings from pymupdf4llm (## or ###)
    md_heading_pattern = r'(?=\n#{1,3} )'

    sections = re.split(md_heading_pattern, text)

    # If the split produced only 1 chunk, the PDF had no Markdown headings.
    # Fall back to the original ALL-CAPS heuristic.
    if len(sections) <= 1:
        print("  [Splitter] No Markdown headings found, falling back to ALL-CAPS splitter.")
        sections = re.split(r'\n(?=[A-Z][A-Z\s]{4,})', text)

    # Drop empty sections
    sections = [s.strip() for s in sections if s.strip()]
    return sections


def create_hierarchical_chunks(text: str, source_name: str) -> tuple[list[dict], list[dict]]:
    

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n---", "\n\n", "\n", " "]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n", ". ", ";", " ", ""]
    )

    sections = split_into_sections(text)

    parents = []
    children = []

    for sec_idx, section in enumerate(sections):
        parent_texts = parent_splitter.split_text(section)

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = f"{source_name}_S{sec_idx}_P{p_idx}"

            parents.append({
                "parent_id": parent_id,
                "text": p_text,
                "metadata": {
                    "source": source_name,
                    "section_id": sec_idx,
                    "type": "parent"
                }
            })

            child_texts = child_splitter.split_text(p_text)

            for c_idx, c_text in enumerate(child_texts):
                children.append({
                    "text": c_text,
                    "metadata": {
                        "source": source_name,
                        "parent_id": parent_id,
                        "section_id": sec_idx,
                        "chunk_id": c_idx,
                        "type": "child"
                    }
                })

    return parents, children


def process_all_pdfs_hierarchical() -> tuple[list[dict], list[dict]]:
    all_parents = []
    all_children = []

    if not os.path.exists(RAW_DATA_DIR):
        print(f"[Chunker] RAW_DATA_DIR not found: {RAW_DATA_DIR}")
        return all_parents, all_children

    pdf_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[Chunker] No PDF files found in {RAW_DATA_DIR}")
        return all_parents, all_children

    for filename in pdf_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        print(f"\n[Chunker] Processing: {filename}...")

        raw_text = extract_text_from_pdf(filepath)

        # Sanity check: warn if extraction returned very little text
        if len(raw_text.strip()) < 500:
            print(f"  [WARNING] Very little text extracted from {filename}. "
                  f"The PDF may be scanned/image-only. Consider nougat for this file.")
            continue

        parents, children = create_hierarchical_chunks(raw_text, filename)
        all_parents.extend(parents)
        all_children.extend(children)

        print(f"  → {len(parents)} parents, {len(children)} children from {filename}")

    print(f"\n[Chunker] Total Parents: {len(all_parents)}")
    print(f"[Chunker] Total Children: {len(all_children)}")

    return all_parents, all_children


# ── Test block ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parents, children = process_all_pdfs_hierarchical()

    print("\n--- SAMPLE CHILD CHUNKS ---")
    for i in range(min(5, len(children))):
        print("\n", "=" * 50)
        print(children[i]["metadata"])
        print(children[i]["text"][:300])

    print("\n--- SAMPLE PARENT ---")
    if parents:
        print(parents[0]["metadata"])
        print(parents[0]["text"][:500])

    # Math spot-check: search for chunks that contain common math symbols
    # to verify extraction quality
    math_symbols = ["^", "α", "β", "θ", "∑", "∇", "²", "³"]
    math_chunks = [
        c for c in children
        if any(sym in c["text"] for sym in math_symbols)
    ]
    print(f"\n--- MATH QUALITY CHECK ---")
    print(f"Chunks containing math symbols: {len(math_chunks)} / {len(children)}")
    if math_chunks:
        print("\nSample math chunk:")
        print(math_chunks[0]["text"])
