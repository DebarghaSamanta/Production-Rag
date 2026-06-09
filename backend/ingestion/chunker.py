import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import RAW_DATA_DIR


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def split_into_sections(text):
    """
    Basic section splitter using uppercase headings.
    Not perfect, but works well for research papers.
    """
    sections = re.split(r'\n(?=[A-Z][A-Z\s]{4,})', text)
    return sections


def create_hierarchical_chunks(text, source_name):
    """
    Returns:
    - parents: list of parent chunks (stored separately)
    - children: list of child chunks (to be embedded)
    """

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
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

            # store parent separately
            parents.append({
                "parent_id": parent_id,
                "text": p_text,
                "metadata": {
                    "source": source_name,
                    "section_id": sec_idx,
                    "type": "parent"
                }
            })

            # create child chunks
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


def process_all_pdfs_hierarchical():
    all_parents = []
    all_children = []

    if not os.path.exists(RAW_DATA_DIR):
        return all_parents, all_children

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(RAW_DATA_DIR, filename)
            print(f"Processing: {filename}...")

            raw_text = extract_text_from_pdf(filepath)
            parents, children = create_hierarchical_chunks(raw_text, filename)

            all_parents.extend(parents)
            all_children.extend(children)

    print(f"Total Parents: {len(all_parents)}")
    print(f"Total Children: {len(all_children)}")

    return all_parents, all_children


# Test block
if __name__ == "__main__":
    parents, children = process_all_pdfs_hierarchical()

    print("\n--- SAMPLE CHILD CHUNKS ---")
    for i in range(min(5, len(children))):
        print("\n", "="*50)
        print(children[i]["metadata"])
        print(children[i]["text"][:300])

    print("\n--- SAMPLE PARENT ---")
    if parents:
        print(parents[0]["metadata"])
        print(parents[0]["text"][:500])