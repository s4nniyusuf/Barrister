def format_docs(docs):
    return "\n\n".join(
        [
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        ]
    )