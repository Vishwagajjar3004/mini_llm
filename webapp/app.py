"""
app.py
======
The Flask backend for the Chat + Document/Code Q&A web app. Ties together:
  - pdf_processor.py   (extract + split text from PDFs/ZIPs; dispatches
                         code files to code_analyzer.py)
  - code_analyzer.py    (structural analysis of uploaded code files)
  - retriever.py         (TF-IDF: find the most relevant sentences)
  - llm_engine.py         (MiniLLM: style/continue the top sentence)
  - storage.py             (persists uploads to disk so they survive restarts)

ROUTES:
  GET  /                 -> serves the frontend page
  POST /api/upload        -> accepts a PDF/ZIP/code file, processes + trains,
                              saves to disk, returns a doc_id
  POST /api/ask            -> accepts {doc_id, question}, returns an answer
  GET  /api/documents       -> lists every previously uploaded document
  POST /api/select           -> loads a previously uploaded document back
                                 into memory by doc_id (no re-upload needed)
  DELETE /api/documents/<id>  -> removes a stored document
  POST /api/chat                -> general chat with the base MiniLLM (no
                                    document needed)

STATE: an in-memory dictionary (DOCUMENTS) caches loaded documents for
fast repeat access, backed by storage.py which persists everything to
disk under webapp/storage/ -- so uploads survive a server restart.
"""

import os
import re
import uuid

from flask import Flask, request, jsonify, render_template
"""
Flask is the web framework.

Functions used:

Flask → creates website
request → receives data
jsonify → returns JSON
render_template → opens HTML page
"""
from pdf_processor import process_uploaded_file
from retriever import Retriever
from llm_engine import DocumentLLM
import storage

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.txt"
)

SUMMARY_KEYWORDS = {"summarize", "summary", "overview", "about", "describe",
                     "explain", "explanation", "structure", "analyze", "analysis"}

# In-memory cache: doc_id -> {"retriever", "llm", "filename", "doc_type", ...}
# Backed by storage.py on disk, so this cache can always be rebuilt.
DOCUMENTS = {}

# ----------------------------------------------------------------------
# The "Chat" tab uses a general-purpose MiniLLM, trained once at startup
# on the project's base dataset (data/dataset.txt) -- this is the exact
# same model chat.py trains via src/train.py, just running inside the
# web app instead of the command line.
# ----------------------------------------------------------------------
BASE_LLM = None


def get_base_llm() -> DocumentLLM:
    """
    Lazily trains (once) and caches the general-purpose chat model.
    Kept separate from per-document models so uploading a file never
    affects the general chat, and vice versa.
    dataset.txt
    Read sentences
    Train MiniLLM
    Save into BASE_LLM
    Return model
    """
    
    global BASE_LLM
    if BASE_LLM is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        BASE_LLM = DocumentLLM()
        BASE_LLM.train_on_sentences(sentences, epochs=60)
    return BASE_LLM


def get_document(doc_id: str):
    """
    Returns the in-memory document dict for `doc_id`, loading it from
    disk (storage.py) into the cache first if needed. Returns None if
    it doesn't exist anywhere (or if doc_id wasn't provided at all).
    """
    if not doc_id:
        return None

    if doc_id in DOCUMENTS:
        return DOCUMENTS[doc_id]

    try:
        loaded = storage.load_document(doc_id)
    except FileNotFoundError:
        return None

    DOCUMENTS[doc_id] = loaded
    return loaded


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    doc_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")
    file.save(save_path)

    try:
        # Step 1: extract + split text/structure (PDF, ZIP, or code file)
        result = process_uploaded_file(save_path, file.filename)

        # Step 2: build the TF-IDF retriever over ALL sentences (cheap,
        # so it scales fine even for longer documents/collections)
        retriever = Retriever(result["all_sentences"])

        # Step 3: train a small MiniLLM on a capped subset of sentences
        # (kept small so training finishes in a few seconds)
        llm = DocumentLLM()
        train_stats = llm.train_on_sentences(result["training_sentences"], epochs=30)

    except ValueError as e:
        os.remove(save_path)
        return jsonify({"error": str(e)}), 400
    finally:
        # We only needed the file on disk long enough to extract text.
        if os.path.exists(save_path):
            os.remove(save_path)

    doc_entry = {
        "retriever": retriever,
        "llm": llm,
        "filename": file.filename,
        "doc_type": result["doc_type"],
        "source_files": result["source_files"],
        "skipped_files": result.get("skipped_files", []),
        "code_explanation": result.get("code_explanation"),
        "num_sentences": result["num_sentences"],
    }
    DOCUMENTS[doc_id] = doc_entry

    # Step 4: persist to disk so this upload survives a server restart.
    storage.save_document(
        doc_id=doc_id,
        filename=file.filename,
        doc_type=result["doc_type"],
        sentences=result["all_sentences"],
        llm=llm,
        source_files=result["source_files"],
        skipped_files=result.get("skipped_files", []),
        code_explanation=result.get("code_explanation"),
    )

    return jsonify(
        {
            "doc_id": doc_id,
            "filename": file.filename,
            "doc_type": result["doc_type"],
            "source_files": result["source_files"],
            "skipped_files": result.get("skipped_files", []),
            "num_sentences": result["num_sentences"],
            "trained_on_sentences": len(result["training_sentences"]),
            "vocab_size": train_stats["vocab_size"],
            "final_loss": round(train_stats["final_loss"], 4),
            "code_explanation": result.get("code_explanation"),
        }
    )


@app.route("/api/documents", methods=["GET"])
def documents():
    """Lists every previously uploaded document (metadata only, no models loaded)."""
    return jsonify({"documents": storage.list_documents()})


@app.route("/api/select", methods=["POST"])
def select():
    """
    Loads a previously uploaded document back into memory by doc_id, so
    the user can keep asking it questions without re-uploading.
    """
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")

    if not doc_id:
        return jsonify({"error": "No doc_id provided."}), 400

    doc = get_document(doc_id)
    if doc is None:
        return jsonify({"error": "That document could not be found."}), 404

    return jsonify(
        {
            "doc_id": doc_id,
            "filename": doc["filename"],
            "doc_type": doc["doc_type"],
            "source_files": doc["source_files"],
            "skipped_files": doc["skipped_files"],
            "num_sentences": doc["num_sentences"],
            "code_explanation": doc.get("code_explanation"),
        }
    )


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    DOCUMENTS.pop(doc_id, None)
    removed = storage.delete_document(doc_id)
    if not removed:
        return jsonify({"error": "That document could not be found."}), 404
    return jsonify({"deleted": True})


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please type a question."}), 400

    doc = get_document(doc_id)
    if doc is None:
        return jsonify({"error": "Unknown document. Please upload a file first."}), 400

    # Special case: for code files, a broad "explain this code" style
    # question gets the full structural explanation directly, since
    # that's a better answer than any single retrieved sentence.
    # We only treat a question as "generic" (wanting a full overview)
    # if, after removing summary keywords and filler words, almost no
    # specific content words are left -- e.g. "explain this code" is
    # generic, but "explain the tokenizer" is a specific question and
    # should still go through normal retrieval below.
    FILLER_WORDS = {"this", "the", "is", "of", "a", "an", "what", "me",
                     "give", "code", "document", "file", "it", "in", "that"}
    # Use regex word extraction (not .split()) so trailing punctuation
    # like "about?" is correctly matched against "about" -- a naive
    # whitespace split would treat "about?" as a different word entirely
    # and silently fail to recognize an otherwise-generic question.
    question_words = set(re.findall(r"[a-z0-9']+", question.lower()))
    remaining_words = question_words - SUMMARY_KEYWORDS - FILLER_WORDS
    is_summary_style = bool(question_words & SUMMARY_KEYWORDS) and len(remaining_words) == 0

    if doc["doc_type"] in ("code", "zip") and is_summary_style and doc.get("code_explanation"):
        return jsonify(
            {
                "extracted_answer": doc["code_explanation"],
                "styled_answer": None,
                "matches": [],
                "is_full_explanation": True,
            }
        )

    # For PDFs/ZIPs, a broad "summarize / what is this about" question
    # doesn't have one right answer to search for -- searching for the
    # literal word "summarize" against the document would just match
    # whichever sentence happens to share the most unrelated words.
    # Instead, give a genuine overview: the first few sentences, which
    # for most documents (reports, articles) capture the opening context.
    if doc["doc_type"] in ("pdf", "zip") and is_summary_style:
        overview_sentences = doc["retriever"].sentences[:5]
        return jsonify(
            {
                "extracted_answer": " ".join(overview_sentences),
                "styled_answer": None,
                "matches": [],
                "is_full_explanation": True,
            }
        )

    # STEP 1 (accurate): retrieve the most relevant real sentences.
    # We ask for more than 1 and combine them, since a single sentence
    # often isn't enough context for a good answer.
    matches = doc["retriever"].query(question, top_k=5)

    if not matches:
        return jsonify(
            {
                "extracted_answer": None,
                "styled_answer": None,
                "matches": [],
                "message": "Couldn't find anything related to that question. "
                           "Try rephrasing, or ask about a specific topic covered in the upload.",
            }
        )

    # Combine the top few genuinely relevant matches (score above a
    # small threshold) into a fuller answer instead of just one sentence.
    strong_matches = [m for m in matches if m["score"] >= matches[0]["score"] * 0.6][:3]
    combined_answer = " ".join(m["sentence"] for m in strong_matches)

    # STEP 2 (styled): let the document-trained MiniLLM continue from
    # the top matching sentence, in the document's own vocabulary.
    try:
        styled_answer = doc["llm"].generate_from_seed(matches[0]["sentence"], max_new_tokens=15)
    except Exception:
        styled_answer = None

    return jsonify(
        {
            "extracted_answer": combined_answer,
            "styled_answer": styled_answer,
            "matches": matches,
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    General chat endpoint (Chat tab) -- no document required. This is
    the same MiniLLM continuation behavior as the command-line chat.py,
    just served over HTTP so it can run entirely from the browser.
    """
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please type a message."}), 400

    llm = get_base_llm()

    # Warn honestly about words the model has never seen (same idea as
    # chat.py's find_unknown_words), instead of silently emitting <UNK>.
    words = llm.tokenizer._split_into_words(message)
    unknown_words = [w for w in words if w not in llm.tokenizer.word2id]

    response = llm.generate_from_seed(message, max_new_tokens=20, temperature=0.8, top_k=10)

    return jsonify({"response": response, "unknown_words": unknown_words})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host=host, port=port)

