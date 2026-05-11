"""Enhanced GDELT matcher with P0+P1+P2 improvements.

P0: Embedding cosine similarity filtering (event title vs article title)
P1: Cross-event URL deduplication (each URL → most relevant event)
P2: Knowledge-graph-expanded entity search terms

Pipeline:
  1. P2: Get KG-expanded entity labels → search DuckDB for candidate articles
  2. P0: Compute embedding similarity between event and each article title
  3. P0: Filter by similarity threshold
  4. P1: Global dedup: each URL assigned to event with highest similarity
  5. Aggregate hourly news_volume
"""

import json
import time
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from config import DATA_DIR


# ---------------------------------------------------------------------------
# Embedding model (lazy load)
# ---------------------------------------------------------------------------
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [Embedding] Loading all-MiniLM-L6-v2 ...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# P2: KG-expanded search terms
# ---------------------------------------------------------------------------
# Entity aliases (same as before)
LABEL_ALIASES = {
    "federal reserve system": ["fed", "federal reserve"],
    "people's republic of china": ["china", "beijing"],
    "united states": ["united states", "u.s."],
    "russia": ["russia", "moscow", "kremlin"],
    "ukraine": ["ukraine", "kyiv"],
    "israel": ["israel", "jerusalem"],
    "iran": ["iran", "tehran"],
    "taiwan": ["taiwan", "taipei"],
    "donald trump": ["trump", "donald trump"],
    "united kingdom": ["uk", "britain"],
}

STOP_WORDS = {"will", "the", "a", "an", "in", "by", "of", "to", "on", "at",
              "or", "and", "for", "is", "be", "has", "have", "do", "does",
              "end", "march", "april", "before", "after", "who", "what", "x",
              "?", "...", "by", "2026", "2027", "31", "30", "25", "2025"}


class EnhancedGDELTMatcher:
    """GDELT news matcher with embedding filtering, dedup, and KG-expanded terms."""

    def __init__(self, graph, db_path=None, sim_threshold=0.35):
        self.graph = graph
        self.db_path = db_path or str(DATA_DIR / "gdelt_master.duckdb")
        self.sim_threshold = sim_threshold
        self.node_index = {n["qid"]: n for n in graph["nodes"]}
        self.url_assignments = {}  # P1: url -> (event_id, max_similarity)

        # Extract event titles for embedding
        self.event_texts = {}
        for evt in graph.get("_events_raw", []):
            eid = evt.get("event_id", "")
            title = evt.get("title", "")
            desc = evt.get("description", "")
            self.event_texts[eid] = f"{title} {desc}"[:256]

    # ---- P2: KG-expanded search terms ----
    def get_expanded_search_terms(self, event_id):
        """Get search terms from KG entities (primary + 1-hop + 2-hop related)."""
        qids = self.graph["event_entity_index"].get(event_id, [])
        terms = set()

        # Primary entities
        for qid in qids:
            node = self.node_index.get(qid)
            if node and node.get("label"):
                terms.add(node["label"].lower())

        # 1-hop and 2-hop related entities via KG edges
        related_qids = set()
        for edge in self.graph["edges"]:
            src, tgt = edge["source"], edge["target"]
            if src in qids:
                related_qids.add(tgt)
            elif tgt in qids:
                related_qids.add(src)

        for qid in related_qids:
            node = self.node_index.get(qid)
            if node and node.get("label"):
                terms.add(node["label"].lower())

        # Title keywords
        evt_text = self.event_texts.get(event_id, "")
        for word in evt_text.lower().replace("?", "").replace(".", "").split():
            word = word.strip()
            if len(word) > 3 and word not in STOP_WORDS:
                terms.add(word)

        # Expand with aliases
        expanded = set(terms)
        for term in terms:
            if term in LABEL_ALIASES:
                expanded.update(LABEL_ALIASES[term])

        return expanded

    # ---- Candidate retrieval from DuckDB ----
    def retrieve_candidates(self, event_id):
        """Query DuckDB with KG-expanded search terms, return candidate articles."""
        search_terms = self.get_expanded_search_terms(event_id)

        if not search_terms:
            return pd.DataFrame(), search_terms

        clauses = []
        for term in sorted(search_terms):
            safe = term.replace("'", "''")
            if len(safe) >= 2:
                clauses.append(f"Estimated_Title ILIKE '%{safe}%'")

        if not clauses:
            return pd.DataFrame(), search_terms

        where = " OR ".join(clauses)
        query = f"""
            SELECT GKGRECORDID, DATE, DocumentIdentifier, Estimated_Title, V2Tone
            FROM gkg_news
            WHERE ({where})
        """

        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            df = conn.execute(query).df()
            conn.close()
        except Exception as e:
            print(f"  [GDELT] Query error: {e}")
            return pd.DataFrame(), search_terms

        return df, search_terms

    # ---- P0: Embedding similarity filtering ----
    def compute_similarities(self, event_id, df):
        """Compute cosine similarity between event text and each article title."""
        if df.empty:
            return df

        event_text = self.event_texts.get(event_id, "")
        if not event_text:
            df["similarity"] = 0.0
            return df

        model = get_embedding_model()

        # Encode event
        event_emb = model.encode([event_text], convert_to_numpy=True)

        # Encode article titles in batches
        titles = df["Estimated_Title"].fillna("").tolist()
        batch_size = 512
        all_sims = []

        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            batch_embs = model.encode(batch, convert_to_numpy=True)
            # Cosine similarity
            sims = np.dot(batch_embs, event_emb.T).flatten()
            sims = sims / (np.linalg.norm(batch_embs, axis=1) * np.linalg.norm(event_emb) + 1e-8)
            all_sims.extend(sims)

        df = df.copy()
        df["similarity"] = all_sims
        return df

    # ---- P1: Cross-event deduplication ----
    def assign_best_event(self, url, event_id, similarity):
        """Keep the event with highest similarity for each URL."""
        if url not in self.url_assignments:
            self.url_assignments[url] = (event_id, similarity)
            return True
        _, best_sim = self.url_assignments[url]
        if similarity > best_sim:
            self.url_assignments[url] = (event_id, similarity)
            return True
        return False

    def resolve_dedup(self, per_event_articles):
        """Second pass: remove articles that belong to a different event."""
        # Build index: url -> best_event
        resolved = {}
        for event_id, articles in per_event_articles.items():
            clean = []
            for a in articles:
                url = a.get("url", "")
                if not url:
                    clean.append(a)
                    continue
                best_eid, _ = self.url_assignments.get(url, (None, 0))
                if best_eid == event_id:
                    clean.append(a)
                # else: this URL was assigned to another event, drop it
            resolved[event_id] = clean
        return resolved

    # ---- Main match flow ----
    def match_event(self, event_id, filter_sim=True):
        """Full match pipeline: candidate retrieval → embedding filter → dedup.

        Returns: dict with articles, search_terms, stats
        """
        df, search_terms = self.retrieve_candidates(event_id)
        if df.empty:
            return {
                "event_id": event_id,
                "articles": [],
                "search_terms": sorted(search_terms),
                "total_retrieved": 0,
                "total_filtered": 0,
            }

        n_retrieved = len(df)

        if filter_sim:
            print(f"    Computing embeddings for {n_retrieved} articles ...")
            t0 = time.time()
            df = self.compute_similarities(event_id, df)
            df = df[df["similarity"] >= self.sim_threshold]
            print(f"    Embedding done in {time.time() - t0:.1f}s, kept {len(df)}/{n_retrieved}")
            n_filtered = len(df)
        else:
            df["similarity"] = 0.0
            n_filtered = n_retrieved

        articles = []
        for _, row in df.iterrows():
            tone_str = str(row.get("V2Tone", ""))
            tone_val = None
            if tone_str:
                parts = tone_str.split(",")
                try:
                    tone_val = float(parts[0])
                except ValueError:
                    tone_val = None

            articles.append({
                "gkg_id": row["GKGRECORDID"],
                "date": str(row["DATE"]),
                "url": row.get("DocumentIdentifier", ""),
                "title": row.get("Estimated_Title", ""),
                "tone": tone_val,
                "similarity": float(row["similarity"]),
            })

        # P1: First-pass URL registration
        for a in articles:
            url = a.get("url", "")
            if url:
                self.assign_best_event(url, event_id, a["similarity"])

        return {
            "event_id": event_id,
            "articles": articles,
            "search_terms": sorted(search_terms),
            "total_retrieved": n_retrieved,
            "total_filtered": n_filtered,
        }

    def compute_hourly_series(self, articles):
        """Aggregate articles to hourly news_volume."""
        if not articles:
            return pd.DataFrame(columns=["datetime_utc", "news_volume"])

        df = pd.DataFrame(articles)
        df["date_str"] = df["date"].astype(str)
        df["hour"] = df["date_str"].str[:10]
        df["datetime_utc"] = pd.to_datetime(df["hour"], format="%Y%m%d%H", errors="coerce")

        hourly = df.groupby("datetime_utc").agg(
            news_volume=("gkg_id", "count"),
        ).reset_index()
        hourly = hourly.sort_values("datetime_utc")
        return hourly


# ---------------------------------------------------------------------------
# Run full alignment for all events
# ---------------------------------------------------------------------------

def run_enhanced_alignment(graph_path=None, output_dir=None, sim_threshold=0.35):
    """Run the full enhanced pipeline for all events in the knowledge graph."""
    from gdelt_matcher import load_graph_from_file

    graph = load_graph_from_file(graph_path)

    # Ensure event texts are set
    entities_path = DATA_DIR / "entities.json"
    if entities_path.exists():
        with open(entities_path) as f:
            entities_data = json.load(f)
        graph["_events_raw"] = entities_data

    output_dir = Path(output_dir) if output_dir else Path("result/enhanced_alignment")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENHANCED GDELT ALIGNMENT (P0+P1+P2)")
    print("=" * 60)
    print(f"  Events: {graph['stats']['total_events']}")
    print(f"  KG nodes: {graph['stats']['total_nodes']}")
    print(f"  KG edges: {graph['stats']['total_edges']}")
    print(f"  Similarity threshold: {sim_threshold}")
    print()

    matcher = EnhancedGDELTMatcher(graph, sim_threshold=sim_threshold)

    # Phase 1: Match each event (P2 + P0), collecting URL assignments (P1)
    phase1_results = {}
    for event_id in sorted(graph["event_entity_index"].keys()):
        event_title = ""
        for evt in graph.get("_events_raw", []):
            if evt.get("event_id") == event_id:
                event_title = evt.get("title", event_id)
                break

        print(f"\n  Event {event_id}: {event_title[:70]}")
        result = matcher.match_event(event_id)
        phase1_results[event_id] = result

        print(f"    Search terms: {result['search_terms'][:8]}...")
        print(f"    Retrieved: {result['total_retrieved']:,}")
        print(f"    After embedding filter: {result['total_filtered']:,}")
        avg_sim = np.mean([a["similarity"] for a in result["articles"]]) if result["articles"] else 0
        print(f"    Mean similarity: {avg_sim:.3f}")

    # Phase 2: Resolve P1 dedup (remove articles assigned to other events)
    print(f"\n{'=' * 60}")
    print("P1 DEDUP: Resolving URL assignments")
    print("=" * 60)
    per_event_raw = {eid: r["articles"] for eid, r in phase1_results.items()}
    per_event_dedup = matcher.resolve_dedup(per_event_raw)

    for event_id in sorted(per_event_dedup.keys()):
        raw_n = len(per_event_raw[event_id])
        dedup_n = len(per_event_dedup[event_id])
        removed = raw_n - dedup_n
        print(f"  {event_id}: {raw_n:,} → {dedup_n:,} ({removed:,} removed)")

    # Phase 3: Save CSVs
    print(f"\n{'=' * 60}")
    print("Saving merged series")
    print("=" * 60)

    # Extract price data from existing CSVs
    import glob
    existing_dir = Path("result/by_date/2026-04-23/qwen_initial_political_events_style_plots")

    results = []
    for event_id in sorted(per_event_dedup.keys()):
        articles = per_event_dedup[event_id]
        hourly = matcher.compute_hourly_series(articles)

        event_title = ""
        for evt in graph.get("_events_raw", []):
            if evt.get("event_id") == event_id:
                event_title = evt.get("title", event_id)
                break

        safe_name = event_title.lower().replace(" ", "-")[:80]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "" for c in safe_name)

        # Try to merge with price
        price_df = None
        pattern = f"event_{event_id}_*_merged_series.csv"
        if existing_dir.exists():
            for f in existing_dir.glob(f"event_{event_id}_*"):
                if f.name.endswith("_merged_series.csv"):
                    try:
                        pdf = pd.read_csv(f)
                        pdf["datetime_utc"] = pd.to_datetime(pdf["datetime_utc"], utc=True)
                        price_df = pdf[["datetime_utc", "price"]].copy()
                        break
                    except Exception:
                        pass

        if price_df is not None and not hourly.empty:
            price_df["hour"] = price_df["datetime_utc"].dt.tz_localize(None).dt.floor("h")
            hourly["hour"] = hourly["datetime_utc"].dt.tz_localize(None).dt.floor("h")
            merged = pd.merge(price_df, hourly, on="hour", how="outer")
            merged["datetime_utc"] = pd.to_datetime(merged["hour"], utc=True)
            merged = merged.drop(columns=["hour"])
            merged = merged.sort_values("datetime_utc")
            merged["news_volume"] = merged["news_volume"].fillna(0).astype(int)
            merged["price"] = merged["price"].interpolate(method="linear", limit_direction="both")
            merged["price"] = merged["price"].ffill().bfill()
        elif not hourly.empty:
            merged = hourly.copy()
            merged["price"] = np.nan
        else:
            continue

        csv_path = output_dir / f"event_{event_id}_{safe_name}_merged_series.csv"
        merged.to_csv(csv_path, index=False)
        print(f"  {event_id}: {len(merged)} rows → {csv_path}")

        results.append({
            "event_id": event_id,
            "event_title": event_title,
            "csv": str(csv_path),
            "rows": len(merged),
            "news_total": len(articles),
            "search_terms": phase1_results[event_id]["search_terms"],
        })

    # Save summary
    summary = []
    for r in results:
        summary.append({
            "event_id": r["event_id"],
            "event_title": r["event_title"],
            "news_total": r["news_total"],
            "rows": r["rows"],
            "search_terms": ", ".join(r["search_terms"][:5]),
        })
    pd.DataFrame(summary).to_csv(output_dir / "alignment_summary.csv", index=False)

    # Print comparison with old
    print(f"\n{'=' * 60}")
    print("COMPARISON: Old vs Enhanced")
    print("=" * 60)
    print(f"{'Event':<45s} {'Old':>8s} {'Enhanced':>8s} {'Reduc%':>8s}")
    print("-" * 75)
    old_counts = {
        "118172": 109748, "236884": 185116, "237306": 185431, "257313": 174959,
        "34044": 18054, "34050": 17127, "35908": 105827, "67284": 59269,
        "73130": 171643, "75478": 17684,
    }
    for r in results:
        eid = r["event_id"]
        old = old_counts.get(eid, 0)
        new = r["news_total"]
        reduc = (1 - new / old) * 100 if old > 0 else 0
        print(f"{r['event_title']:<45s} {old:>8,} {new:>8,} {reduc:>7.1f}%")

    return results


if __name__ == "__main__":
    import sys
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    out = sys.argv[2] if len(sys.argv) > 2 else "result/by_date/2026-05-11/kg_gdelt_alignment_v2"
    run_enhanced_alignment(output_dir=out, sim_threshold=threshold)
