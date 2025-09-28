import os
import re
import json
import argparse
import sqlite3
import numpy as np

# sentence-transformers is the simplest way to get BGE/E5 locally for a prototype
from sentence_transformers import SentenceTransformer

# PyPI package that bundles the sqlite-vec extension and a helper loader
import sqlite_vec

# Dynamic parser imports based on filename
import importlib

# --------------------
# Helpers
# --------------------


def get_parser_module(html_path):
    """
    Determine which parser module to use based on the HTML filename.
    Returns the appropriate parser module.
    """
    filename = os.path.basename(html_path)
    base_name = os.path.splitext(filename)[0]  # Remove .html extension

    try:
        parser_module = importlib.import_module(f"parse.parse_{base_name}")
        return parser_module
    except ImportError:
        raise ValueError(
            f"No parser found for {filename}. Expected parse_{base_name}.py"
        )


def read_rules(html_path):
    """
    Read rules from HTML file using the appropriate parser.
    Returns: list of dicts {number, title, body}
    """
    items = []

    # Get the appropriate parser module
    parser = get_parser_module(html_path)

    # Load and parse the rules HTML
    rules_tree = parser.load_rules(html_path=html_path)

    # Flatten the hierarchical structure
    flat_rules = parser.flatten(rules_tree)

    # Convert each rule node to the expected format
    for rule_node in flat_rules:
        # Skip rules without meaningful content
        if not rule_node.get("title") and not rule_node.get("id"):
            continue

        # Use label if available, otherwise use id, otherwise create a number
        number = rule_node.get("label") or rule_node.get("id") or "Unknown"

        # Use the title as the main title
        title = rule_node.get("title", "").strip()
        if not title:
            title = number

        # Build body from available content
        body_parts = []

        # Add the title to body if it exists and is different from number
        if title and title != number:
            body_parts.append(title)

        # Add annotations if they exist
        annotations = rule_node.get("annotations", [])
        if annotations:
            for annotation in annotations:
                if annotation.strip():
                    body_parts.append(f"Note: {annotation.strip()}")

        # Add tooltip information if it exists
        tooltips = rule_node.get("tooltips", [])
        if tooltips:
            for tooltip in tooltips:
                tooltip_text = tooltip.get("text", "").strip()
                tooltip_title = tooltip.get("title", "").strip()
                if tooltip_text:
                    if tooltip_title:
                        body_parts.append(f"Tooltip: {tooltip_text} ({tooltip_title})")
                    else:
                        body_parts.append(f"Tooltip: {tooltip_text}")

        # Add note if it exists (from table parsing)
        note = rule_node.get("note")
        if note:
            body_parts.append(f"Additional: {note}")

        # Create the final body
        body = "\n".join(body_parts) if body_parts else title

        items.append({"number": number, "title": title, "body": body})

    return items


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def chunk_text(text, max_words=220, overlap_words=40):
    """
    Fast+simple chunker: sentence-aware if possible, then packs sentences.
    """
    if not text:
        return []
    sents = _SENT_SPLIT.split(text.strip())
    chunks, cur, cur_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur_words + w > max_words and cur:
            chunks.append(" ".join(cur).strip())
            # overlap
            if overlap_words > 0 and chunks[-1]:
                tail = chunks[-1].split()[-overlap_words:]
                cur = [" ".join(tail)]
                cur_words = len(cur[0].split())
            else:
                cur, cur_words = [], 0
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def ensure_db(conn):
    cur = conn.cursor()
    # metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rules(
          rule_id INTEGER PRIMARY KEY,
          number  TEXT,
          title   TEXT,
          body    TEXT
        );
    """)
    # FTS5 (contentless)
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
          number, title, body, content=''
        );
    """)
    # Vector table; use cosine distance
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS rules_vec USING vec0(
          emb float[384] distance_metric=cosine
        );
    """)
    conn.commit()


def insert_rule(conn, number, title, body, rowid=None):
    cur = conn.cursor()
    if rowid is None:
        cur.execute(
            "INSERT INTO rules(number,title,body) VALUES (?,?,?)", (number, title, body)
        )
        rowid = cur.lastrowid
    else:
        cur.execute(
            "INSERT INTO rules(rule_id,number,title,body) VALUES (?,?,?,?)",
            (rowid, number, title, body),
        )
    # FTS5: set rowid so it matches rules rowid
    cur.execute(
        "INSERT INTO rules_fts(rowid, number, title, body) VALUES (?,?,?,?)",
        (rowid, number, title, body),
    )
    return rowid


def insert_embedding(conn, rowid, emb_vec):
    """
    emb_vec: 1D array/list of length 384 (float32). We store via vec_f32(json) for simplicity.
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO rules_vec(rowid, emb) VALUES (?, vec_f32(?))",
        (rowid, json.dumps([float(x) for x in emb_vec])),
    )


def encode_passages(model, texts):
    # BGE recommends "passage: " prefix for documents
    prefixed = [f"passage: {t}" for t in texts]
    vecs = model.encode(
        prefixed, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    return vecs.astype(np.float32)


def encode_queries(model, texts):
    # BGE recommends "query: " prefix for queries
    prefixed = [f"query: {t}" for t in texts]
    vecs = model.encode(
        prefixed, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    return vecs.astype(np.float32)


def bm25(conn, q, k=50):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid AS rule_id, bm25(rules_fts) AS score
        FROM rules_fts
        WHERE rules_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """,
        (q, k),
    )
    return [(r[0], float(r[1])) for r in cur.fetchall()]


def knn(conn, q_emb, k=20):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rowid AS rule_id, distance
        FROM rules_vec
        WHERE emb MATCH vec_f32(?)
        AND k = ?
        ORDER BY distance
    """,
        (json.dumps([float(x) for x in q_emb]), k),
    )
    # Convert distance (1 - cosine) into a similarity-ish score to align directions (higher better)
    out = []
    for rid, dist in cur.fetchall():
        sim = 1.0 - float(dist)  # cosine_sim = 1 - distance
        out.append((rid, sim))
    return out


def rrf_fuse(bm25_ranks, knn_ranks, c=60, top_n=5):
    """
    bm25_ranks: list of (id, score) lower score is better in bm25()
      We'll convert to ranks
    knn_ranks: list of (id, score) higher score is better (similarity)
    """
    # turn into rank positions
    bm_ids = [rid for rid, _ in bm25_ranks]
    kn_ids = [rid for rid, _ in knn_ranks]

    bm_pos = {rid: i + 1 for i, rid in enumerate(bm_ids)}
    kn_pos = {rid: i + 1 for i, rid in enumerate(kn_ids)}

    scores = {}
    for rid in set(bm_ids) | set(kn_ids):
        s = 0.0
        if rid in bm_pos:
            s += 1.0 / (c + bm_pos[rid])
        if rid in kn_pos:
            s += 1.0 / (c + kn_pos[rid])
        scores[rid] = s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [rid for rid, _ in ranked]


def fetch_rules(conn, ids):
    if not ids:
        return []
    qmarks = ",".join("?" * len(ids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT rule_id, number, title, body
        FROM rules
        WHERE rule_id IN ({qmarks})
    """,
        ids,
    )
    m = {rid: (num, tit, bod) for rid, num, tit, bod in cur.fetchall()}
    return [(rid,) + m[rid] for rid in ids if rid in m]


def build_or_open_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  # loads the bundled vec0 extension
    ensure_db(conn)
    return conn


def ingest_corpus(conn, items, model, chunk_words=220, overlap=40):
    """
    items: list of dicts {number,title,body}
    For each rule item, chunk its body; each chunk becomes its own row (keeps same number/title with suffix).
    """
    chunk_texts = []
    meta = []
    for it in items:
        chunks = chunk_text(it["body"], max_words=chunk_words, overlap_words=overlap)
        if not chunks:
            chunks = [it["body"]]
        for idx, ch in enumerate(chunks, start=1):
            number = f"{it['number']}#{idx}" if len(chunks) > 1 else it["number"]
            title = it["title"]
            body = ch
            meta.append((number, title, body))
            chunk_texts.append(body)

    # Insert rows
    rowids = []
    cur = conn.cursor()
    for number, title, body in meta:
        cur.execute(
            "INSERT INTO rules(number,title,body) VALUES (?,?,?)", (number, title, body)
        )
        rid = cur.lastrowid
        # FTS
        cur.execute(
            "INSERT INTO rules_fts(rowid, number, title, body) VALUES (?,?,?,?)",
            (rid, number, title, body),
        )
        rowids.append(rid)
    conn.commit()

    # Embeddings in batches
    vecs = encode_passages(model, chunk_texts)
    for rid, emb in zip(rowids, vecs):
        insert_embedding(conn, rid, emb)
    conn.commit()


def search(conn, model, q, k=20, top_n=5, c=60):
    q_vec = encode_queries(model, [q])[0]
    bm = bm25(conn, q, k=max(k, 50))
    kn = knn(conn, q_vec, k=k)
    fused_ids = rrf_fuse(bm, kn, c=c, top_n=top_n)
    results = fetch_rules(conn, fused_ids)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid search for rules - pass .html to rebuild or .db to use existing"
    )
    ap.add_argument(
        "file",
        help="Either .html file (to rebuild database) or .db file (to use existing database)",
    )
    ap.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument(
        "--k", type=int, default=20, help="K for vector/BM25 candidate pools"
    )
    ap.add_argument("--top_n", type=int, default=5, help="Final results to show")
    args = ap.parse_args()

    # Determine if we're rebuilding based on file extension
    file_path = args.file
    if file_path.endswith(".html"):
        # Rebuild mode: HTML file provided
        html_file = file_path
        db_file = os.path.splitext(file_path)[0] + ".db"
        rebuild = True

        if not os.path.exists(html_file):
            print(f"HTML file not found: {html_file}")
            return

        print(f"Rebuild mode: {html_file} → {db_file}")

    elif file_path.endswith(".db"):
        # Use existing database mode
        db_file = file_path
        html_file = None
        rebuild = False

        if not os.path.exists(db_file):
            print(f"Database file not found: {db_file}")
            print("If you want to create a new database, pass the .html file instead.")
            return

        print(f"Using existing database: {db_file}")

    else:
        print("File must be either .html (to rebuild) or .db (to use existing)")
        return

    conn = build_or_open_db(db_file)

    if rebuild:
        # Clear existing data by dropping and recreating tables
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS rules;")
        cur.execute("DROP TABLE IF EXISTS rules_fts;")
        cur.execute("DROP TABLE IF EXISTS rules_vec;")
        conn.commit()

        # Recreate the schema
        ensure_db(conn)

        print(f"Loading rules from {html_file}...")
        items = read_rules(html_file)
        if not items:
            print("No rules found in HTML file.")
            return
        print(f"Ingesting {len(items)} rules...")
        model = SentenceTransformer(args.model)
        ingest_corpus(conn, items, model)
    else:
        # Check if database has data
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rules")
        n = cur.fetchone()[0]
        if n == 0:
            print("Database is empty. Pass an .html file to rebuild it.")
            return
        print(f"Database loaded with {n} chunks.")
        model = SentenceTransformer(args.model)

    # Quick interactive loop
    print("\nType a query (or 'exit')")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        results = search(conn, model, q, k=args.k, top_n=args.top_n)
        print(f"Top {len(results)}")
        for rid, number, title, body in results:
            snippet = (body[:240] + "…") if len(body) > 240 else body
            print(f"- [{number}] {title}\\n  {snippet}\\n")


if __name__ == "__main__":
    main()
