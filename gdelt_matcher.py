"""Real GDELT matcher: queries DuckDB for news matching knowledge graph entities.

Replaces the mock GDELTMatcher with DuckDB-powered keyword search.
Matches Polymarket events to GDELT news by searching entity labels in article titles.
Computes hourly news_volume and avg_tone for time series alignment with prices.
"""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from config import DATA_DIR


# Entity label aliases: shorter/common forms for GDELT title search
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
}


class RealGDELTMatcher:
    """Query GDELT news via DuckDB using knowledge graph entity labels."""

    def __init__(self, graph, db_path=None):
        self.graph = graph
        self.db_path = db_path or str(DATA_DIR / "gdelt_master.duckdb")
        self.conn = duckdb.connect(self.db_path, read_only=True)
        self.node_index = {n["qid"]: n for n in graph["nodes"]}

    def _expand_labels(self, labels):
        """Add shorter/common aliases for entity labels."""
        expanded = set(labels)
        for label in labels:
            if label in LABEL_ALIASES:
                expanded.update(LABEL_ALIASES[label])
        return expanded

    def get_entity_qids_for_event(self, event_id):
        return self.graph["event_entity_index"].get(event_id, [])

    # Stop words to skip when extracting title keywords
    STOP_WORDS = {"will", "the", "a", "an", "in", "by", "of", "to", "on", "at",
                  "or", "and", "for", "is", "be", "has", "have", "do", "does",
                  "end", "march", "april", "before", "after", "who", "what", "x",
                  "?", "...", "by", "2026", "2027", "31", "30", "25", "2025"}

    def get_entity_labels_for_event(self, event_id, expand=False):
        """Get entity labels (search terms) for an event from the knowledge graph.

        If expand=False, only uses primary entities (directly matched from event title).
        If expand=True, also includes 1-hop related entities from KG expansion.
        """
        qids = self.get_entity_qids_for_event(event_id)
        labels = set()
        for qid in qids:
            node = self.node_index.get(qid)
            if node:
                label = node["label"]
                if label and len(label) > 1:
                    labels.add(label.lower())

        if expand:
            # Also add labels from related entities (1-hop edges)
            for edge in self.graph["edges"]:
                if edge["source"] in qids or edge["target"] in qids:
                    for q in [edge["source"], edge["target"]]:
                        if q in self.node_index:
                            lbl = self.node_index[q]["label"]
                            if lbl and len(lbl) > 1:
                                labels.add(lbl.lower())

        # Also extract significant title words as supplementary search terms
        for evt in self.graph.get("_events_raw", []):
            if evt.get("event_id") == event_id:
                title = evt.get("title", "")
                for word in title.lower().replace("?", "").replace(".", "").split():
                    word = word.strip()
                    if len(word) > 3 and word not in self.STOP_WORDS:
                        labels.add(word)
                break

        # Expand with aliases for better search recall
        labels = self._expand_labels(labels)
        return labels

    def build_search_query(self, labels):
        """Build a DuckDB SQL query that searches labels in article titles."""
        if not labels:
            return None

        # Build ILIKE conditions for each label
        clauses = []
        for label in sorted(labels):
            # Escape single quotes for SQL
            safe_label = label.replace("'", "''")
            if len(safe_label) >= 2:
                clauses.append(f"Estimated_Title ILIKE '%{safe_label}%'")

        if not clauses:
            return None

        where = " OR ".join(clauses)
        return f"SELECT GKGRECORDID, DATE, Estimated_Title, V2Tone FROM gkg_news WHERE ({where})"

    def match_news(self, event_id, time_window_days=None):
        """Find GDELT news articles matching an event's knowledge graph entities.

        Returns list of article dicts with date, title, tone.
        """
        labels = self.get_entity_labels_for_event(event_id)
        if not labels:
            return {
                "event_id": event_id,
                "matched_entities": [],
                "articles": [],
                "search_terms": [],
            }

        query = self.build_search_query(labels)
        if query is None:
            return {
                "event_id": event_id,
                "matched_entities": [],
                "articles": [],
                "search_terms": sorted(labels),
            }

        try:
            rows = self.conn.execute(query).fetchall()
        except Exception as e:
            print(f"  [GDELT] Query error: {e}")
            rows = []

        articles = []
        for row in rows:
            gkg_id, date_str, title, tone_str = row
            # Parse V2Tone: "tone,pos,neg,polarity,activity,self_ref,wordcount"
            tone_val = None
            if tone_str:
                parts = tone_str.split(",")
                if len(parts) >= 1:
                    try:
                        tone_val = float(parts[0])
                    except ValueError:
                        tone_val = None

            articles.append({
                "gkg_id": gkg_id,
                "date": date_str,
                "title": title,
                "tone": tone_val,
            })

        # Get entity details for the event
        qids = self.get_entity_qids_for_event(event_id)
        entities = [self.node_index[q] for q in qids if q in self.node_index]

        return {
            "event_id": event_id,
            "matched_entities": entities,
            "articles": articles,
            "search_terms": sorted(labels),
            "total_matched": len(articles),
        }

    def compute_hourly_series(self, event_id, start_date=None, end_date=None):
        """Compute hourly news_volume and avg_tone for an event.

        Returns DataFrame with columns: datetime_utc, news_volume, avg_tone
        """
        result = self.match_news(event_id)
        articles = result["articles"]

        if not articles:
            return pd.DataFrame(columns=["datetime_utc", "news_volume", "avg_tone"])

        df = pd.DataFrame(articles)
        df["date_str"] = df["date"].astype(str)

        # Parse GDELT date format: YYYYMMDDHHMMSS -> hour bucket
        df["hour"] = df["date_str"].str[:10]  # YYYYMMDDHH
        df["datetime_utc"] = pd.to_datetime(df["hour"], format="%Y%m%d%H", errors="coerce")

        # Filter by date range if specified
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df["datetime_utc"] >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df["datetime_utc"] <= end]

        # Aggregate by hour
        hourly = df.groupby("datetime_utc").agg(
            news_volume=("gkg_id", "count"),
            avg_tone=("tone", "mean"),
        ).reset_index()

        hourly = hourly.sort_values("datetime_utc")
        return hourly

    def merge_with_price(self, event_id, price_data, start_date=None, end_date=None):
        """Merge hourly GDELT news series with Polymarket price data.

        price_data: list of {timestamp, price} dicts from Polymarket.
        Returns DataFrame with columns: datetime_utc, price, news_volume, avg_tone
        """
        news_df = self.compute_hourly_series(event_id, start_date, end_date)

        # Build price DataFrame
        price_df = pd.DataFrame(price_data)
        if price_df.empty:
            return news_df

        if "t" in price_df.columns:
            price_df["datetime_utc"] = pd.to_datetime(price_df["t"], unit="s", utc=True)
        elif "timestamp" in price_df.columns:
            price_df["datetime_utc"] = pd.to_datetime(price_df["timestamp"], utc=True)
        elif "ts" in price_df.columns:
            price_df["datetime_utc"] = pd.to_datetime(price_df["ts"], unit="s", utc=True)
        else:
            return news_df

        # Floor to hour for alignment
        price_df["hour"] = price_df["datetime_utc"].dt.floor("h")
        news_df["hour"] = news_df["datetime_utc"].dt.floor("h")

        # Average price within each hour
        price_hourly = price_df.groupby("hour")["price"].mean().reset_index()

        # Merge
        merged = pd.merge(price_hourly, news_df, on="hour", how="outer")
        merged["datetime_utc"] = merged["hour"]
        merged = merged.drop(columns=["hour"])
        merged = merged.sort_values("datetime_utc")

        # Fill missing values
        merged["news_volume"] = merged["news_volume"].fillna(0).astype(int)
        merged["price"] = merged["price"].interpolate(method="linear")
        merged["price"] = merged["price"].fillna(method="ffill").fillna(method="bfill")

        return merged[["datetime_utc", "price", "news_volume", "avg_tone"]]

    def close(self):
        self.conn.close()


def load_graph_from_file(path=None):
    """Load knowledge graph from JSON file."""
    import json
    if path is None:
        path = DATA_DIR / "knowledge_graph.json"
    with open(path) as f:
        return json.load(f)


def load_price_curves(path=None):
    """Load Polymarket price curves from JSON."""
    import json
    if path is None:
        path = DATA_DIR / "price_curves.json"
    with open(path) as f:
        return json.load(f)


def get_event_price_data(curves, event_id):
    """Extract price history for a given event from curves data."""
    event_curves = [c for c in curves if c.get("event_id") == event_id]
    history = []
    for curve in event_curves:
        for h in curve.get("history", []):
            history.append(h)
    return history


def run_gdelt_alignment(graph_path=None, curves_path=None, output_dir=None):
    """Full GDELT alignment pipeline: match news, merge with prices, save CSV."""
    graph = load_graph_from_file(graph_path)
    curves = load_price_curves(curves_path)
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    matcher = RealGDELTMatcher(graph)

    results = []
    for event_id in graph["event_entity_index"]:
        print(f"  Processing event: {event_id}")
        price_history = get_event_price_data(curves, event_id)

        if price_history:
            merged = matcher.merge_with_price(event_id, price_history)
        else:
            merged = matcher.compute_hourly_series(event_id)

        event_title = ""
        for e in graph.get("_events_raw", []):
            if e.get("event_id") == event_id:
                event_title = e.get("title", "")
                break

        safe_name = event_title.lower().replace(" ", "-")[:80]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "" for c in safe_name)

        csv_path = output_dir / f"event_{event_id}_{safe_name}_merged_series.csv"
        merged.to_csv(csv_path, index=False)
        print(f"    Saved {len(merged)} rows to {csv_path}")
        results.append({"event_id": event_id, "rows": len(merged), "csv": str(csv_path)})

    matcher.close()
    return results


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "result/gdelt_alignment"
    run_gdelt_alignment(output_dir=output)
