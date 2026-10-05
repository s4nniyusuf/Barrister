import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CORPUS_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def get_corpus_files(corpus_dir: str) -> list:
    """Get all corpus files in the corpus directory."""
    return list(Path(corpus_dir).rglob("*.jsonl"))


def read_corpus_file(file_path: str) -> list:
    """Read a corpus file and return its content in a list."""
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content


def convert_to_langchain_documents(data: list):
    """Convert a list of data to LangChain documents."""
    docs = [
        Document(
            page_content=item["text"],
            metadata={"document_id": item["document_id"], 
                      "source_file": item["source_file"],
                      "record_type": item["record_type"],
                      "physical_page_start": item["physical_page_start"],
                      "physical_page_end": item["physical_page_end"],
                      "chapter": item["chapter"],
                      "chapter_title": item["chapter_title"],
                      "part": item["part"],
                      "part_title": item["part_title"],
                      "section_number": item["section_number"],
                      "section_title": item["section_title"]
                    }
        )
        for item in data
    ]
    return docs


def chunk_text(docs: list):
    """Split langchain documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def convert_chunks_to_dict(chunks: list) -> list:
    """Convert a list of langchain documents to a list of dictionaries."""
    return [
        {               
            "chunk_id": f"chunk_{i:06d}",
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for i, chunk in enumerate(chunks)
    ]


def save_chunks(chunks_dict: list):
    """Save the chunks to the output directory."""
    output_dir = Path(CHUNKS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks_dict:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(chunks_dict)} chunks to {output_path}")


def main():
    """Main function to read and chunk corpus files."""
    corpus_files = get_corpus_files(CORPUS_DIR)
    data = []
    # loop through all corpus files and read their content
    for file in corpus_files:
        content = read_corpus_file(file)
        data.extend(content)
    docs = convert_to_langchain_documents(data)
    chunks = chunk_text(docs)
    chunks_dict = convert_chunks_to_dict(chunks)
    save_chunks(chunks_dict)


if __name__ == "__main__":
    main()