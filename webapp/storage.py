"""
storage.py
==========
Makes uploads PERSISTENT across server restarts.

Without this, the Flask app's `DOCUMENTS` dictionary lives only in
memory -- restarting `python app.py` would forget every upload,
forcing you to re-upload and re-train every time.

WHAT GETS SAVED, PER UPLOAD:
  storage/<doc_id>/
    meta.json     -- filename, doc type, sentences, source files, timestamp
    vocab.json     -- the document's tokenizer vocabulary (via Tokenizer.save)
    model.pt        -- the trained MiniLLM's weights (via DocumentLLM.save)

This is plain files on disk (no database) -- simple, and easy to
inspect/delete manually if you ever want to.
"""

import os
import json
import time

from retriever import Retriever
from llm_engine import DocumentLLM

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)


def _doc_dir(doc_id: str) -> str:
    return os.path.join(STORAGE_DIR, doc_id)


def save_document(doc_id: str, filename: str, doc_type: str, sentences: list,
                   llm: DocumentLLM, source_files: list = None,
                   skipped_files: list = None, code_explanation: str = None) -> None:
    """
    Persists everything needed to fully reconstruct this document's
    retriever + trained model later, without re-uploading or re-training.
    """
    dir_path = _doc_dir(doc_id)
    os.makedirs(dir_path, exist_ok=True)

    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": doc_type,  # "pdf" | "zip" | "code"
        "sentences": sentences,
        "source_files": source_files or [filename],
        "skipped_files": skipped_files or [],
        "code_explanation": code_explanation,
        "uploaded_at": time.time(),
    }
    with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    llm.save(dir_path)


def load_document(doc_id: str) -> dict:
    """
    Reconstructs a document's retriever + trained MiniLLM from disk.
    Raises FileNotFoundError if this doc_id was never saved.
    """
    dir_path = _doc_dir(doc_id)
    meta_path = os.path.join(dir_path, "meta.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No stored document found for doc_id={doc_id}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    retriever = Retriever(meta["sentences"])
    llm = DocumentLLM.load(dir_path)

    return {
        "retriever": retriever,
        "llm": llm,
        "filename": meta["filename"],
        "doc_type": meta["doc_type"],
        "source_files": meta["source_files"],
        "skipped_files": meta["skipped_files"],
        "code_explanation": meta.get("code_explanation"),
        "num_sentences": len(meta["sentences"]),
    }


def list_documents() -> list:
    """
    Returns metadata (NOT the full model) for every document ever
    uploaded, newest first -- used to populate a "previously uploaded"
    list in the UI without loading every model into memory.
    """
    documents = []
    if not os.path.isdir(STORAGE_DIR):
        return documents

    for doc_id in os.listdir(STORAGE_DIR):
        meta_path = os.path.join(STORAGE_DIR, doc_id, "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            documents.append({
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "doc_type": meta["doc_type"],
                "num_sentences": len(meta["sentences"]),
                "uploaded_at": meta["uploaded_at"],
            })
        except (json.JSONDecodeError, KeyError):
            continue  # skip corrupted entries rather than crashing

    documents.sort(key=lambda d: d["uploaded_at"], reverse=True)
    return documents


def delete_document(doc_id: str) -> bool:
    """Removes a stored document from disk. Returns True if it existed."""
    import shutil
    dir_path = _doc_dir(doc_id)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        return True
    return False
