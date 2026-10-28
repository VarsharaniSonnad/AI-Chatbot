"""
main.py
-------
RAG chatbot backend using OpenAI's GPT API.

Put your already-scraped website content into ONE of these:
  - documents/content.txt   (plain text file — simplest option)
  - documents/pages.json    (list of {"url","title","text"} objects)

Setup:
    pip install flask flask-cors openai numpy python-dotenv
    echo "OPENAI_API_KEY=your-openai-key-here" > .env

Run:
    python main.py
Then open index.html in your browser (it calls this server at localhost:5000).

Caching: embeddings are saved to embeddings_cache.pkl after the first run.
On future runs, if your document hasn't changed, the cache is reused and
NO embedding API calls are made — startup is instant. If you edit
documents/content.txt (or pages.json), the change is detected automatically
and the index is rebuilt. Delete embeddings_cache.pkl to force a rebuild.
"""
import hashlib
import json
import os
import pickle
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
# =========================
# CHAT LOGGING
# =========================
# File where chat logs are saved
LOG_FILE = "chat_logs.jsonl"

# Create a logger for chat interactions
chat_logger = logging.getLogger("chat_logger")

# Set log level to INFO
chat_logger.setLevel(logging.INFO)

# Write logs to the chat_logs.jsonl file
chat_handler = logging.FileHandler(LOG_FILE)

# Save only the message without extra log details
chat_handler.setFormatter(logging.Formatter("%(message)s"))

# Connect the file handler to the logger
chat_logger.addHandler(chat_handler)

# Prevent duplicate logs in the terminal
chat_logger.propagate = False


# Function to save one chat interaction
def log_interaction(question, answer, sources, response_time_ms):

    # Store chat details in a dictionary
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # Current time
        "question": question,                                  # User question
        "answer": answer,                                      # Bot answer
        "sources": sources,                                    # Retrieved sources
        "response_time_ms": response_time_ms,                  # Response time
    }

    # Convert to JSON and save as one line in the log file
    chat_logger.info(json.dumps(entry, ensure_ascii=False))
# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in a .env file first.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# 2. MODEL AND RAG SETTINGS
# ============================================================

EMBED_MODEL_NAME = "./models/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
SALES_CONTACT_INFO = "+91-8019176868" 
DOCS_DIR = Path("documents")
CACHE_FILE = Path("embeddings_cache.pkl")

app = Flask(__name__)
CORS(app)  # allow the HTML page (any origin) to call this API — restrict before going live


# ---------- 1. Load your scraped document(s) ----------
def load_source_pages() -> list[dict]:
    txt_path = DOCS_DIR / "content.txt"

    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
        return [{
            "url": "local-document",
            "title": "Alphaleo Content",
            "text": text
        }]

    raise FileNotFoundError(
        "No document found. Put your scraped content in documents/content.txt"
    )

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0

    # Continue creating chunks until the entire document is processed
    while start < len(text):

        # Create one chunk from the current document
        chunks.append(text[start : start + size])

        # Move forward by (chunk size - overlap)
        # With size=800 and overlap=150, next chunk starts at 650
        start += size - overlap

    # Remove empty chunks and extra spaces, then return all chunks
    return [c.strip() for c in chunks if c.strip()]


# ---------- 2. Build the index at startup (cached to disk) ----------
def chunk_hash(text: str) -> str:
    # Convert a chunk into a unique SHA-256 hash
    # The same exact chunk text always produces the same hash
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


print("Loading document(s)...")

# Load the CURRENT document from documents/content.txt
pages = load_source_pages()

# ============================================================
# CREATE CHUNKS AND HASHES FOR THE CURRENT DOCUMENT
# ============================================================
# Build current chunk list with a hash per chunk
current_chunk_records = []
for page in pages:
    # Create chunks from the CURRENT document
    for c in chunk_text(page.get("text", ""), CHUNK_SIZE, CHUNK_OVERLAP):
        # Store the current chunk and its hash
        current_chunk_records.append({
            "text": c,
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            # Create a hash for this CURRENT chunk
            "hash": chunk_hash(c),
        })

# ============================================================
# LOAD OLD CHUNKS + EMBEDDINGS FROM embeddings_cache.pkl
# ============================================================
old_embeddings_by_hash = {}
# Check whether embeddings_cache.pkl already exists
if CACHE_FILE.exists():
    print("Found cached embeddings...")
    # Open and load the old cache file
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    # Reuse the cache only if the same embedding model was used
    if cache.get("model") == EMBED_MODEL_NAME:
        # Get old chunk records from the cache
        # Each record contains text, URL, title, and hash
        old_chunk_records = cache.get("chunk_records", [])
        # Get old embeddings from the cache
        old_embeddings = cache.get("embeddings", [])
        # Connect each old chunk with its corresponding embedding
        # Position 0 chunk -> Position 0 embedding
        for rec, embedding in zip(old_chunk_records, old_embeddings):
            # Check that the old chunk record contains a hash
            # OLD CHUNK HASH -> OLD EMBEDDING
            if "hash" in rec:
                old_embeddings_by_hash[rec["hash"]] = embedding
        print(f"Loaded {len(old_embeddings_by_hash)} cached chunk embeddings.")
    else:
        # If the embedding model changed, don't reuse old embeddings
        print("Embedding model changed — all embeddings will be recreated.")
# ============================================================
# COMPARE CURRENT CHUNK HASHES WITH CACHED CHUNK HASHES
# ============================================================
# Reuse unchanged chunks, embed only new/changed ones
embeddings_list = []
reused_count = 0
new_count = 0
# Go through every chunk from the CURRENT document
for rec in current_chunk_records:
    # Get the hash of the CURRENT chunk
    h = rec["hash"]
    if h in old_embeddings_by_hash:
        # YES: This exact chunk already existed before
        # Reuse its old embedding from embeddings_cache.pkl
        vector = old_embeddings_by_hash[h]
        reused_count += 1
    else:
        # NO: This is a new or changed chunk
        # Create a NEW embedding using the local embedding model
        vector = embed_model.encode(rec["text"]).tolist()
        new_count += 1
    # Add either the reused or newly created embedding
    # to the final embedding list
    embeddings_list.append(vector)

chunk_records = current_chunk_records
embeddings = np.array(embeddings_list)

print(f"Reused {reused_count} embeddings, created {new_count} new embeddings.")

with open(CACHE_FILE, "wb") as f:
    pickle.dump(
        {"model": EMBED_MODEL_NAME, "chunk_records": chunk_records, "embeddings": embeddings},
        f,
    )

print("Index ready.")

# ---------- 3. Retrieval + generation ----------

def retrieve(query: str, k: int = TOP_K):
    # Create embedding for the user's question — local model, no API call
    query_vec = np.array(embed_model.encode(query))

    # cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    # Calculate similarity between question and every chunk
    sims = (embeddings @ query_vec) / np.where(norms == 0, 1e-9, norms)
    # Get indexes of highest similarity scores
    top_idx = np.argsort(sims)[::-1][:k]
    # Return the corresponding document chunks
    return [chunk_records[i] for i in top_idx]
# ============================================================
#  SYSTEM INSTRUCTION FOR GPT
# ============================================================

SYSTEM_INSTRUCTION = (
    "You are Alphaleo's AI assistant. Answer using ONLY the provided context. "
    "If asked if you're AI or human, say clearly you are an AI assistant. "
    "Never invent facts, emails, phone numbers, or dates — except the approved "
    "sales contact details given below, which are real and must be used exactly "
    "as provided. For anything else, if you don't have a real detail, point to "
    "alphaleo.tech instead of guessing. "

    "If a question is vague or broad (e.g. 'what services do you provide' with no "
    "specifics), don't dump everything — ask a short clarifying question offering 2-3 "
    "concrete categories from the context to choose from, e.g. 'Are you asking about our "
    "Salesforce consulting, our AI/Vision Bench solutions, or something else?' "

    "If a question asks for something not in the context and unrelated to what Alphaleo "
    "publicly shares (e.g. employee count, salaries, internal financials, personal "
    "opinions, anything off-topic), politely decline and redirect, in this style: "
    "'I cannot help with that. I can assist you with information about Alphaleo's "
    "services, industries we work in, and company background. What would you like to "
    "know?' Keep it short, polite, and don't guess an answer. "

    "When asked broadly what Alphaleo does, start with its core identity (Salesforce "
    "partner / CRM consulting firm) as the first point, before specific service lines "
    "like Copado or Vision Bench AI. "

    "Default format: short bullets — bolded title + one sentence each. Only give longer, "
    "multi-sentence detail if the user asks to 'explain' or 'elaborate'. "
    "Always speak on behalf of Alphaleo using 'we', 'our', and 'us'. "
    "Do not use 'I' or 'my' when referring to Alphaleo's services, "
    "solutions, capabilities, or company information. "

    "Never mention 'provided context', 'context', 'retrieved documents', "
    "'RAG', 'documents', 'source', 'knowledge base', 'chunks', or "
    "'embeddings' in the answer. Answer naturally without explaining "
    "where the information came from. "
    "Be concise and friendly. Don't repeat a greeting/introduction — that's handled "
    "separately in the UI. "

    "SALES HANDOFF RULE (always follow this exactly): If the user asks to talk to "
    "sales or connect with an associate, do not hand off or give contact details "
    "immediately. First ask ONE short clarifying question — what service they're "
    "interested in (Salesforce consulting, AI/Vision Bench, Copado, or something "
    "else) and whether it's a new or existing engagement. Only after they answer — "
    "or if they clearly insist without answering — give this message in full, "
    f"worded naturally, including the phone number{SALES_CONTACT_INFO} and email exactly as written: "
    
)


# ============================================================
#  REWRITE FOLLOW-UP QUESTIONS
# ============================================================
def rewrite_query_if_needed(question: str, history: list) -> str:
    """
    If the question is a follow-up (like 'explain more'), rewrites it into a
    standalone question using recent history. Otherwise returns it unchanged.
    """
    history_text = "\n".join(f"{h['role']}: {h['text']}" for h in history[-6:])
    # Ask GPT to rewrite only if necessary
    prompt = (
        f"History:\n{history_text}\n\n"
        f"Message: \"{question}\"\n\n"
        "If this message already makes sense on its own, reply with it unchanged. "
        "If it's a follow-up (like 'explain more', 'what about that'), rewrite it as a "
        "standalone question using the history. Reply with ONLY the question, nothing else."
    )
    # GPT decides whether rewriting is needed
    result = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return result.choices[0].message.content.strip()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
# ============================================================
#  CHAT API
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    # Get current user question
    question = (data or {}).get("message", "").strip()
    # Get previous conversation
    history = (data or {}).get("history", [])
    # Reject empty questions
    if not question:
        return jsonify({"error": "message is required"}), 400
    MAX_EXCHANGES = 20
    if len(history) >= MAX_EXCHANGES * 2:
        return jsonify({
            "answer": "We've reached the limit for this conversation. Please start a new conversation to continue chatting with me.",
            "limit_reached": True
        })

    # PATH 1: exact repeat question — respond naturally, not by pasting the same answer
    for i, h in enumerate(history):
        if h["role"] == "user" and h["text"].strip().lower() == question.lower():
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                previous_answer = history[i + 1]["text"]
                print("Exact repeat — generating a human-style acknowledgment.")
                followup_prompt = (
                    f"The user already asked this exact question earlier, and you answered:\n"
                    f"\"{previous_answer}\"\n\n"
                    "They just asked the same question again. Respond naturally and briefly, "
                    "like a helpful human would — acknowledge you already covered this "
                    "(e.g. 'As I mentioned' or 'Like I said'), give a short 1-2 sentence "
                    "recap (not the full answer again), then ask if they'd like more detail "
                    "or something else. Keep it warm and short, not repetitive."
                )
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    temperature=0,
                    messages=[{"role": "user", "content": followup_prompt}],
                )
                return jsonify({"answer": response.choices[0].message.content})

    # PATH 2: no history at all — definitely a standalone question, skip rewriting
    if not history:
        search_query = question
        print("No history — using question as-is for retrieval.")
    else:
        # PATH 2 or 3: ask the rewriter to decide AND rewrite in one call
        search_query = rewrite_query_if_needed(question, history)
    # Search the document using the final search query
    matches = retrieve(search_query)
    # Combine retrieved chunks into one text block
    context = "\n\n---\n\n".join(
        f"[Source: {m['title']} ({m['url']})]\n{m['text']}" for m in matches
    )
    # Convert previous messages into readable text
    history_text = "\n".join(f"{h['role'].capitalize()}: {h['text']}" for h in history[-6:])
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    # Send history + RAG context + question to GPT
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )

    answer_text = response.choices[0].message.content
    sources = [{"title": m["title"], "url": m["url"]} for m in matches]
    log_interaction(question, answer_text, sources, 0)

    return jsonify({"answer": answer_text, "sources": sources})


@app.route("/health", methods=["GET"])
def health():
    # Used to check whether backend is running
    return jsonify({"status": "ok", "chunks_indexed": len(chunk_records)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)